#!/bin/bash
# User data script for master node

# Update system
apt-get update -y
apt-get upgrade -y

# Install essential packages
apt-get install -y \
    build-essential \
    cmake \
    git \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libcurl4-openssl-dev \
    htop \
    nfs-kernel-server \
    nfs-common \
    python3 \
    python3-pip \
    awscli

# Set hostname
hostnamectl set-hostname hpc-master
echo "127.0.0.1 hpc-master" >> /etc/hosts

# Create ubuntu user if not exists and set up SSH
if ! id "ubuntu" &>/dev/null; then
    useradd -m -s /bin/bash ubuntu
fi

# Create shared directory
mkdir -p /shared
chown ubuntu:ubuntu /shared
chmod 755 /shared

# Configure NFS server
echo "/shared *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
exportfs -a
systemctl restart nfs-kernel-server
systemctl enable nfs-kernel-server

# Set up MPI environment for ubuntu user
cat >> /home/ubuntu/.bashrc << 'EOF'
export PATH=$PATH:/usr/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/openmpi/lib
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0
EOF

# Generate SSH key for MPI communication
sudo -u ubuntu ssh-keygen -t rsa -N "" -f /home/ubuntu/.ssh/id_rsa
sudo -u ubuntu cp /home/ubuntu/.ssh/id_rsa.pub /home/ubuntu/.ssh/authorized_keys
sudo -u ubuntu chmod 600 /home/ubuntu/.ssh/authorized_keys
sudo -u ubuntu chmod 700 /home/ubuntu/.ssh

# Copy SSH key to shared directory for workers
cp -r /home/ubuntu/.ssh /shared/
chown -R ubuntu:ubuntu /shared/.ssh

# Configure SSH for passwordless login
cat >> /home/ubuntu/.ssh/config << 'EOF'
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    LogLevel=quiet
EOF
chown ubuntu:ubuntu /home/ubuntu/.ssh/config
chmod 600 /home/ubuntu/.ssh/config

# Create hostfile template
cat > /shared/hostfile << 'EOF'
hpc-master slots=1
EOF

# Clone and build the search engine
cd /shared
sudo -u ubuntu git clone https://github.com/PramithaMJ/High-Performance-Parallel-Search-Engine.git
cd High-Performance-Parallel-Search-Engine/Hybrid\ Version/
chown -R ubuntu:ubuntu .

# Create optimized Makefile for t2.medium
cat > Makefile << 'EOF'
CC = mpicc
CXX = mpicxx
CFLAGS = -O2 -fopenmp -march=native -mtune=native -pipe
CXXFLAGS = -O2 -fopenmp -march=native -mtune=native -std=c++11 -pipe
LDFLAGS = -fopenmp -lcurl -lm

SRCDIR = src
OBJDIR = obj
BINDIR = bin
INCDIR = include

SOURCES = $(wildcard $(SRCDIR)/*.c $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/search_engine

$(shell mkdir -p $(OBJDIR) $(BINDIR))

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.c.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

install: $(TARGET)
	mkdir -p /usr/local/bin
	cp $(TARGET) /usr/local/bin/

.PHONY: clean install
EOF

# Build the project
sudo -u ubuntu make clean && sudo -u ubuntu make

# Create run script for t2.medium cluster
cat > /shared/run_cluster.sh << 'EOF'
#!/bin/bash

# Configuration for t2.medium (2 vCPUs, 4GB RAM)
NODES=$(cat /shared/hostfile | wc -l)
MPI_PROCESSES=$NODES
OMP_THREADS=2

echo " Running HPC Search Engine on t2.medium cluster"
echo "Configuration:"
echo "  - Nodes: $NODES"
echo "  - MPI Processes: $MPI_PROCESSES"
echo "  - OpenMP Threads per process: $OMP_THREADS"

# Set environment variables
export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_STACKSIZE=512K

# Run the search engine
mpirun -np $MPI_PROCESSES \
       --hostfile /shared/hostfile \
       --map-by node \
       --bind-to core \
       -x OMP_NUM_THREADS \
       -x OMP_PROC_BIND \
       -x OMP_PLACES \
       -x OMP_STACKSIZE \
       /shared/High-Performance-Parallel-Search-Engine/Hybrid\ Version/bin/search_engine "$@"
EOF

chmod +x /shared/run_cluster.sh
chown ubuntu:ubuntu /shared/run_cluster.sh

# Create monitoring script
cat > /shared/monitor_cluster.sh << 'EOF'
#!/bin/bash

echo " HPC Cluster Monitoring Dashboard"
echo "=================================="

while true; do
    clear
    echo "=== t2.medium Cluster Status $(date) ==="
    echo ""
    
    echo "Node Status:"
    while read -r line; do
        node=$(echo $line | awk '{print $1}')
        if ping -c 1 -W 1 $node > /dev/null 2>&1; then
            echo "   $node: Online"
        else
            echo "  $node: Offline"
        fi
    done < /shared/hostfile
    
    echo ""
    echo "Resource Usage:"
    echo "Master Node:"
    uptime
    free -h | grep Mem
    
    echo ""
    echo "Disk Usage:"
    df -h | grep -E "(/$|/shared)"
    
    echo ""
    echo "Network Connections:"
    ss -tuln | grep :22
    
    sleep 10
done
EOF

chmod +x /shared/monitor_cluster.sh
chown ubuntu:ubuntu /shared/monitor_cluster.sh

# Create simple web dashboard
mkdir -p /shared/dashboard
cat > /shared/dashboard/server.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import subprocess
import json
import threading
import time
from datetime import datetime

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>HPC Cluster Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .online { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .offline { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .metric { background-color: #e2e3e5; padding: 10px; margin: 5px 0; border-radius: 3px; }
    </style>
</head>
<body>
    <h1> HPC Search Engine Cluster Dashboard</h1>
    <p><strong>Last Updated:</strong> {timestamp}</p>
    
    <h2>Cluster Status</h2>
    {cluster_status}
    
    <h2>System Metrics</h2>
    {system_metrics}
    
    <h2>Quick Actions</h2>
    <ul>
        <li><a href="/run">Run Search Engine</a></li>
        <li><a href="/logs">View Logs</a></li>
        <li><a href="/health">Health Check</a></li>
    </ul>
</body>
</html>
"""
            
            # Get cluster status
            try:
                result = subprocess.run(['cat', '/shared/hostfile'], capture_output=True, text=True)
                nodes = result.stdout.strip().split('\n')
                status_html = ""
                for node in nodes:
                    node_name = node.split()[0]
                    ping_result = subprocess.run(['ping', '-c', '1', '-W', '1', node_name], 
                                                capture_output=True)
                    if ping_result.returncode == 0:
                        status_html += f'<div class="status online"> {node_name}: Online</div>'
                    else:
                        status_html += f'<div class="status offline"> {node_name}: Offline</div>'
            except:
                status_html = '<div class="status offline"> Unable to check node status</div>'
            
            # Get system metrics
            try:
                uptime_result = subprocess.run(['uptime'], capture_output=True, text=True)
                memory_result = subprocess.run(['free', '-h'], capture_output=True, text=True)
                disk_result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
                
                metrics_html = f"""
                <div class="metric"><strong>Uptime:</strong> {uptime_result.stdout.strip()}</div>
                <div class="metric"><strong>Memory:</strong><pre>{memory_result.stdout}</pre></div>
                <div class="metric"><strong>Disk:</strong><pre>{disk_result.stdout}</pre></div>
                """
            except:
                metrics_html = '<div class="metric"> Unable to get system metrics</div>'
            
            html = html.format(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cluster_status=status_html,
                system_metrics=metrics_html
            )
            
            self.wfile.write(html.encode())
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health = {"status": "healthy", "timestamp": datetime.now().isoformat()}
            self.wfile.write(json.dumps(health).encode())
        
        else:
            super().do_GET()

if __name__ == "__main__":
    PORT = 8080
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard server running on port {PORT}")
        httpd.serve_forever()
EOF

chmod +x /shared/dashboard/server.py

# Start dashboard service
cat > /etc/systemd/system/hpc-dashboard.service << 'EOF'
[Unit]
Description=HPC Cluster Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/shared/dashboard
ExecStart=/usr/bin/python3 /shared/dashboard/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable hpc-dashboard
systemctl start hpc-dashboard

# Configure automatic updates
echo "unattended-upgrades unattended-upgrades/enable_auto_updates boolean true" | debconf-set-selections
apt-get install -y unattended-upgrades

# Log completion
echo "Master node setup completed at $(date)" >> /var/log/hpc-setup.log
