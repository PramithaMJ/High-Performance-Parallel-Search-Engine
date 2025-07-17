#!/bin/bash

# User data script for MPI Worker Nodes
# This script runs on instance startup to configure worker nodes

set -e

# Variables from Terraform
CLUSTER_NAME="${cluster_name}"
EFS_ID="${efs_id}"
AWS_REGION="${aws_region}"
MPI_VERSION="${mpi_version}"
S3_BUCKET="${s3_bucket}"
MPI_SLOTS_PER_NODE="${mpi_slots_per_node}"
AUTO_SHUTDOWN_ENABLED="${auto_shutdown_enabled}"
IDLE_TIMEOUT_MINUTES="${idle_timeout_minutes}"

# Logging
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting MPI Worker Node setup at $(date)"

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
    bc

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
                        "file_path": "/var/log/mpi-worker.log",
                        "log_group_name": "/aws/ec2/${cluster_name}",
                        "log_stream_name": "worker-{instance_id}"
                    },
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/${cluster_name}",
                        "log_stream_name": "user-data-worker-{instance_id}"
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

# Install OpenMPI (same version as master)
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

# Create MPI user (same as master)
useradd -m -s /bin/bash mpiuser
usermod -aG sudo mpiuser

# Setup SSH for MPI user - will receive public key from master
sudo -u mpiuser mkdir -p /home/mpiuser/.ssh
sudo -u mpiuser chmod 700 /home/mpiuser/.ssh

# Configure SSH for passwordless access
cat > /home/mpiuser/.ssh/config << EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    LogLevel=quiet
EOF
chown mpiuser:mpiuser /home/mpiuser/.ssh/config
chmod 600 /home/mpiuser/.ssh/config

# Mount EFS
mkdir -p /shared
echo "$EFS_ID.efs.$AWS_REGION.amazonaws.com:/ /shared nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,intr,timeo=600 0 0" >> /etc/fstab

# Wait for EFS to be available and mount
for i in {1..30}; do
    if mount -a; then
        echo "EFS mounted successfully"
        break
    else
        echo "Waiting for EFS to be available... attempt $i/30"
        sleep 10
    fi
done

# Wait for master node to complete setup
echo "Waiting for master node setup to complete..."
for i in {1..60}; do
    if [ -f /shared/hostfile ]; then
        echo "Master node setup detected"
        break
    else
        echo "Waiting for master node... attempt $i/60"
        sleep 30
    fi
done

# Copy SSH key from shared storage (will be placed there by master)
while [ ! -f /shared/.ssh/id_rsa.pub ]; do
    echo "Waiting for SSH keys from master..."
    sleep 10
done

sudo -u mpiuser cp /shared/.ssh/id_rsa.pub /home/mpiuser/.ssh/authorized_keys
sudo -u mpiuser chmod 600 /home/mpiuser/.ssh/authorized_keys

# Register this worker node in the hostfile
WORKER_IP=$(hostname -I | awk '{print $1}')
echo "$WORKER_IP slots=$MPI_SLOTS_PER_NODE" >> /shared/hostfile

# Setup performance monitoring
cat > /usr/local/bin/worker-monitor.sh << 'EOF'
#!/bin/bash
# Worker node monitoring script

LOG_FILE="/var/log/mpi-worker.log"
METRICS_FILE="/shared/metrics/worker-$(hostname).json"

mkdir -p /shared/metrics

while true; do
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1)
    
    # Create JSON metrics
    cat > $METRICS_FILE << JSONEOF
{
    "timestamp": "$TIMESTAMP",
    "hostname": "$(hostname)",
    "ip": "$(hostname -I | awk '{print $1}')",
    "cpu_usage": $CPU_USAGE,
    "memory_usage": $MEM_USAGE,
    "disk_usage": $DISK_USAGE,
    "load_average": "$LOAD_AVG",
    "mpi_processes": $(pgrep -c mpirun || echo 0),
    "status": "active"
}
JSONEOF
    
    # Log metrics
    echo "$(date): CPU: $CPU_USAGE%, MEM: $MEM_USAGE%, DISK: $DISK_USAGE%, LOAD: $LOAD_AVG" >> $LOG_FILE
    
    sleep 60
done
EOF

chmod +x /usr/local/bin/worker-monitor.sh

# Create systemd service for monitoring
cat > /etc/systemd/system/worker-monitor.service << EOF
[Unit]
Description=MPI Worker Node Monitor
After=network.target

[Service]
Type=simple
User=mpiuser
ExecStart=/usr/local/bin/worker-monitor.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable worker-monitor
systemctl start worker-monitor

# Setup auto-shutdown if enabled
if [ "$AUTO_SHUTDOWN_ENABLED" = "true" ]; then
    cat > /usr/local/bin/worker-auto-shutdown.sh << EOF
#!/bin/bash
# Auto-shutdown script for idle worker instances

IDLE_THRESHOLD=$IDLE_TIMEOUT_MINUTES
LOG_FILE="/var/log/auto-shutdown.log"

# Check if there are any active MPI processes
MPI_PROCESSES=\$(pgrep -c mpirun || echo 0)

if [ \$MPI_PROCESSES -eq 0 ]; then
    # Check CPU usage
    CPU_USAGE=\$(awk '{u=\$2+\$4; t=\$2+\$3+\$4+\$5; if (NR==1){u1=u; t1=t;} else print (\$2+\$4-u1) * 100 / (t-t1); }' \\
               <(grep 'cpu ' /proc/stat; sleep 1; grep 'cpu ' /proc/stat))
    
    if (( \$(echo "\$CPU_USAGE < 5" | bc -l) )); then
        echo "\$(date): No MPI processes and low CPU usage (\$CPU_USAGE%), initiating shutdown..." >> \$LOG_FILE
        
        # Remove from hostfile
        WORKER_IP=\$(hostname -I | awk '{print \$1}')
        grep -v "\$WORKER_IP" /shared/hostfile > /tmp/hostfile.tmp && mv /tmp/hostfile.tmp /shared/hostfile
        
        # Mark as shutting down
        echo '{"status": "shutting_down", "timestamp": "'"\$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"'"}' > /shared/metrics/worker-\$(hostname).json
        
        shutdown -h +2 "Auto-shutdown due to idle time"
    else
        echo "\$(date): No MPI processes but CPU usage: \$CPU_USAGE%" >> \$LOG_FILE
    fi
else
    echo "\$(date): \$MPI_PROCESSES MPI processes running" >> \$LOG_FILE
fi
EOF

    chmod +x /usr/local/bin/worker-auto-shutdown.sh
    
    # Add to cron to check every 15 minutes
    echo "*/15 * * * * root /usr/local/bin/worker-auto-shutdown.sh" >> /etc/crontab
fi

# Setup network optimization for MPI
cat >> /etc/sysctl.conf << EOF
# Network optimizations for MPI
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 2500
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_congestion_control = bbr
EOF
sysctl -p

# Create worker health check script
cat > /usr/local/bin/health-check.sh << 'EOF'
#!/bin/bash
# Health check script for worker node

# Check if essential services are running
if ! systemctl is-active --quiet worker-monitor; then
    echo "UNHEALTHY: worker-monitor service not running"
    exit 1
fi

# Check if EFS is mounted
if ! mountpoint -q /shared; then
    echo "UNHEALTHY: EFS not mounted"
    exit 1
fi

# Check if MPI is working
if ! which mpirun > /dev/null; then
    echo "UNHEALTHY: MPI not available"
    exit 1
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEM_USAGE -gt 95 ]; then
    echo "UNHEALTHY: High memory usage ($MEM_USAGE%)"
    exit 1
fi

echo "HEALTHY"
exit 0
EOF

chmod +x /usr/local/bin/health-check.sh

# Signal that worker setup is complete
touch /var/log/worker-setup-complete
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws s3 cp /var/log/user-data.log s3://$S3_BUCKET/logs/worker-$INSTANCE_ID-$(date +%Y%m%d-%H%M%S).log

echo "MPI Worker Node setup completed at $(date)"
echo "Worker IP: $(hostname -I | awk '{print $1}')"
echo "Available slots: $MPI_SLOTS_PER_NODE"
