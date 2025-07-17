#!/bin/bash

# User data script for MPI Master Node
# This script runs on instance startup to configure the master node

set -e

# Variables from Terraform
CLUSTER_NAME="${cluster_name}"
EFS_ID="${efs_id}"
AWS_REGION="${aws_region}"
MPI_VERSION="${mpi_version}"
S3_BUCKET="${s3_bucket}"
WEB_PORT="${web_port}"
ENABLE_WEB_INTERFACE="${enable_web_interface}"
AUTO_SHUTDOWN_ENABLED="${auto_shutdown_enabled}"
IDLE_TIMEOUT_MINUTES="${idle_timeout_minutes}"

# Logging
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting MPI Master Node setup at $(date)"

# Update system
apt-get update -y
apt-get upgrade -y

# Install essential packages
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    awscli \
    nfs-common \
    libcurl4-openssl-dev \
    pkg-config \
    python3 \
    python3-pip \
    jq \
    unzip

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/mpi-search-engine.log",
                        "log_group_name": "/aws/ec2/${cluster_name}",
                        "log_stream_name": "master-{instance_id}"
                    },
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/${cluster_name}",
                        "log_stream_name": "user-data-{instance_id}"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "MPI/SearchEngine",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Install OpenMPI
cd /tmp
MPI_MAJOR_VERSION=$(echo $MPI_VERSION | cut -d. -f1-2)
wget https://download.open-mpi.org/release/open-mpi/v$MPI_MAJOR_VERSION/openmpi-$MPI_VERSION.tar.gz
tar -xzf openmpi-$MPI_VERSION.tar.gz
cd openmpi-$MPI_VERSION

./configure --prefix=/usr/local --enable-mpi-cxx
make -j$(nproc)
make install

# Update library path
echo "/usr/local/lib" >> /etc/ld.so.conf.d/openmpi.conf
ldconfig

# Add MPI to PATH
echo 'export PATH=/usr/local/bin:$PATH' >> /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> /etc/environment
source /etc/environment

# Create MPI user
useradd -m -s /bin/bash mpiuser
usermod -aG sudo mpiuser

# Setup SSH for MPI user
sudo -u mpiuser ssh-keygen -t rsa -N "" -f /home/mpiuser/.ssh/id_rsa
sudo -u mpiuser cp /home/mpiuser/.ssh/id_rsa.pub /home/mpiuser/.ssh/authorized_keys
sudo -u mpiuser chmod 600 /home/mpiuser/.ssh/authorized_keys
sudo -u mpiuser chmod 700 /home/mpiuser/.ssh

# Configure SSH for passwordless access
cat >> /home/mpiuser/.ssh/config << EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    LogLevel=quiet
EOF
chown mpiuser:mpiuser /home/mpiuser/.ssh/config

# Mount EFS
mkdir -p /shared
echo "$EFS_ID.efs.$AWS_REGION.amazonaws.com:/ /shared nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600 0 0" >> /etc/fstab
mount -a

# Set permissions for shared directory
chown mpiuser:mpiuser /shared
chmod 755 /shared

# Create application directories
mkdir -p /shared/mpi-search-engine
mkdir -p /shared/dataset
mkdir -p /shared/results
mkdir -p /shared/logs
chown -R mpiuser:mpiuser /shared

# Download and compile the search engine
cd /shared/mpi-search-engine
git clone https://github.com/your-username/mpi-search-engine.git . || {
    # If git clone fails, download from S3
    aws s3 cp s3://$S3_BUCKET/source/mpi-search-engine.tar.gz /tmp/
    tar -xzf /tmp/mpi-search-engine.tar.gz -C /shared/mpi-search-engine --strip-components=1
}

# Build the application
make clean
make all

# Download sample dataset if available
if [ -n "$DATASET_S3_BUCKET" ]; then
    aws s3 sync s3://$DATASET_S3_BUCKET/dataset/ /shared/dataset/
fi

# Create hostfile
cat > /shared/hostfile << EOF
# MPI Hostfile - will be updated by cluster setup script
$(hostname -I | awk '{print $1}') slots=4
EOF

# Install web interface dependencies if enabled
if [ "$ENABLE_WEB_INTERFACE" = "true" ]; then
    pip3 install flask flask-cors boto3 psutil
    
    # Create web interface
    cat > /shared/web_interface.py << 'WEBEOF'
#!/usr/bin/env python3
import os
import subprocess
import json
import time
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import boto3
import psutil

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return {'status': 'healthy', 'timestamp': time.time()}

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MPI Search Engine Cluster</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { background: #f0f0f0; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .search-box { margin: 20px 0; }
            .search-box input { width: 60%; padding: 10px; }
            .search-box button { padding: 10px 20px; background: #007cba; color: white; border: none; }
            .results { margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MPI Search Engine Cluster</h1>
            <div class="status">
                <h3>Cluster Status</h3>
                <div id="cluster-status">Loading...</div>
            </div>
            <div class="search-box">
                <h3>Search</h3>
                <input type="text" id="query" placeholder="Enter search query...">
                <button onclick="search()">Search</button>
            </div>
            <div class="results" id="results" style="display:none;">
                <h3>Results</h3>
                <div id="search-results"></div>
            </div>
        </div>
        <script>
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('cluster-status').innerHTML = 
                            `<strong>Nodes:</strong> ${data.nodes}<br>
                             <strong>Total CPUs:</strong> ${data.total_cpus}<br>
                             <strong>Memory Usage:</strong> ${data.memory_percent}%<br>
                             <strong>CPU Usage:</strong> ${data.cpu_percent}%`;
                    });
            }
            
            function search() {
                const query = document.getElementById('query').value;
                if (!query) return;
                
                document.getElementById('results').style.display = 'block';
                document.getElementById('search-results').innerHTML = 'Searching...';
                
                fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('search-results').innerHTML = 
                        `<strong>Execution Time:</strong> ${data.execution_time}s<br>
                         <strong>Results Found:</strong> ${data.results.length}<br>
                         <pre>${JSON.stringify(data.results, null, 2)}</pre>`;
                });
            }
            
            updateStatus();
            setInterval(updateStatus, 30000);
        </script>
    </body>
    </html>
    ''')

@app.route('/api/status')
def api_status():
    # Get cluster status
    status = {
        'nodes': 1,  # Will be updated by cluster script
        'total_cpus': psutil.cpu_count(),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_percent': psutil.cpu_percent(),
        'timestamp': time.time()
    }
    return jsonify(status)

@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    start_time = time.time()
    
    # Run MPI search
    try:
        cmd = [
            'mpirun', '--hostfile', '/shared/hostfile',
            '/shared/mpi-search-engine/bin/search_engine',
            '-q', query
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results
            results = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    results.append(line.strip())
            
            return jsonify({
                'query': query,
                'results': results,
                'execution_time': round(execution_time, 2),
                'status': 'success'
            })
        else:
            return jsonify({
                'error': result.stderr,
                'status': 'error'
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'error': 'Search timeout',
            'status': 'timeout'
        }), 408
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
WEBEOF

    chmod +x /shared/web_interface.py
    chown mpiuser:mpiuser /shared/web_interface.py
fi

# Create cluster management scripts
cat > /shared/cluster-setup.sh << 'EOF'
#!/bin/bash
# Cluster setup script to run after all nodes are ready

# Update hostfile with all nodes
echo "Updating hostfile with cluster nodes..."
echo "# MPI Hostfile - Auto-generated" > /shared/hostfile

# Master node
echo "$(hostname -I | awk '{print $1}') slots=4" >> /shared/hostfile

# Worker nodes will be added by the deployment script
EOF

cat > /shared/run-search.sh << 'EOF'
#!/bin/bash
# Script to run distributed search

QUERY="$1"
if [ -z "$QUERY" ]; then
    echo "Usage: $0 'search query'"
    exit 1
fi

echo "Running MPI search for: $QUERY"
echo "Cluster nodes:"
cat /shared/hostfile

mpirun --hostfile /shared/hostfile \
       --map-by node \
       --bind-to core \
       /shared/mpi-search-engine/bin/search_engine -q "$QUERY"
EOF

chmod +x /shared/*.sh
chown mpiuser:mpiuser /shared/*.sh

# Setup auto-shutdown if enabled
if [ "$AUTO_SHUTDOWN_ENABLED" = "true" ]; then
    cat > /usr/local/bin/auto-shutdown.sh << EOF
#!/bin/bash
# Auto-shutdown script for idle instances

IDLE_THRESHOLD=$IDLE_TIMEOUT_MINUTES
LOG_FILE="/var/log/auto-shutdown.log"

# Check CPU usage over the last $IDLE_THRESHOLD minutes
CPU_USAGE=\$(awk '{u=\$2+\$4; t=\$2+\$3+\$4+\$5; if (NR==1){u1=u; t1=t;} else print (\$2+\$4-u1) * 100 / (t-t1); }' \\
           <(grep 'cpu ' /proc/stat; sleep 1; grep 'cpu ' /proc/stat))

if (( \$(echo "\$CPU_USAGE < 5" | bc -l) )); then
    echo "\$(date): Low CPU usage (\$CPU_USAGE%), initiating shutdown..." >> \$LOG_FILE
    # Notify SNS topic if configured
    if [ -n "\$SNS_TOPIC_ARN" ]; then
        aws sns publish --topic-arn \$SNS_TOPIC_ARN \\
            --message "Auto-shutdown initiated for instance \$(curl -s http://169.254.169.254/latest/meta-data/instance-id)" \\
            --region $AWS_REGION
    fi
    shutdown -h +5 "Auto-shutdown due to idle time"
else
    echo "\$(date): CPU usage: \$CPU_USAGE%" >> \$LOG_FILE
fi
EOF

    chmod +x /usr/local/bin/auto-shutdown.sh
    
    # Add to cron to check every 10 minutes
    echo "*/10 * * * * root /usr/local/bin/auto-shutdown.sh" >> /etc/crontab
fi

# Start web interface if enabled
if [ "$ENABLE_WEB_INTERFACE" = "true" ]; then
    cat > /etc/systemd/system/mpi-web.service << EOF
[Unit]
Description=MPI Search Engine Web Interface
After=network.target

[Service]
Type=simple
User=mpiuser
WorkingDirectory=/shared
ExecStart=/usr/bin/python3 /shared/web_interface.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    systemctl enable mpi-web
    systemctl start mpi-web
fi

# Signal that setup is complete
touch /var/log/master-setup-complete
aws s3 cp /var/log/user-data.log s3://$S3_BUCKET/logs/master-$(date +%Y%m%d-%H%M%S).log

echo "MPI Master Node setup completed at $(date)"
