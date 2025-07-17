#!/bin/bash
# User data script for worker nodes

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
    nfs-common \
    python3 \
    python3-pip \
    awscli

# Set hostname
hostnamectl set-hostname hpc-worker-${worker_index}
echo "127.0.0.1 hpc-worker-${worker_index}" >> /etc/hosts

# Create ubuntu user if not exists
if ! id "ubuntu" &>/dev/null; then
    useradd -m -s /bin/bash ubuntu
fi

# Wait for master node to be ready
echo "Waiting for master node to be ready..."
while ! ping -c 1 -W 5 ${master_ip} > /dev/null 2>&1; do
    echo "Master node not ready, waiting..."
    sleep 10
done

# Create shared directory mount point
mkdir -p /shared
chown ubuntu:ubuntu /shared

# Mount shared directory from master
echo "Mounting shared directory from master..."
while ! mount -t nfs ${master_ip}:/shared /shared; do
    echo "Failed to mount NFS, retrying..."
    sleep 10
done

# Add to fstab for persistent mount
echo "${master_ip}:/shared /shared nfs defaults,_netdev 0 0" >> /etc/fstab

# Set up MPI environment for ubuntu user
cat >> /home/ubuntu/.bashrc << 'EOF'
export PATH=$PATH:/usr/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/openmpi/lib
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0
EOF

# Wait for SSH keys to be available in shared directory
echo "Waiting for SSH keys from master..."
while [ ! -f /shared/.ssh/id_rsa ]; do
    echo "SSH keys not ready, waiting..."
    sleep 5
done

# Copy SSH keys from shared directory
sudo -u ubuntu cp -r /shared/.ssh /home/ubuntu/
chown -R ubuntu:ubuntu /home/ubuntu/.ssh
chmod 700 /home/ubuntu/.ssh
chmod 600 /home/ubuntu/.ssh/*

# Configure SSH for passwordless login
cat >> /home/ubuntu/.ssh/config << 'EOF'
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    LogLevel=quiet
EOF
chown ubuntu:ubuntu /home/ubuntu/.ssh/config
chmod 600 /home/ubuntu/.ssh/config

# Add this worker to the hostfile
echo "hpc-worker-${worker_index} slots=1" >> /shared/hostfile

# Update /etc/hosts with all nodes
cat >> /etc/hosts << EOF
${master_ip} hpc-master
EOF

# Get worker IPs and add to hosts file
# Note: This is simplified - in production you'd get actual IPs
WORKER_IP=$(hostname -I | awk '{print $1}')
echo "$WORKER_IP hpc-worker-${worker_index}" >> /etc/hosts

# Create worker monitoring script
cat > /home/ubuntu/monitor_worker.sh << 'EOF'
#!/bin/bash

echo "📊 Worker Node Monitor"
echo "===================="

while true; do
    clear
    echo "=== Worker Node Status $(date) ==="
    echo ""
    
    echo "System Info:"
    echo "  Hostname: $(hostname)"
    echo "  IP: $(hostname -I | awk '{print $1}')"
    echo "  Uptime: $(uptime -p)"
    
    echo ""
    echo "Resource Usage:"
    echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% used"
    echo "  Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    echo "  Load: $(uptime | awk -F'load average:' '{print $2}')"
    
    echo ""
    echo "Network:"
    echo "  Master connection: $(ping -c 1 -W 1 hpc-master > /dev/null 2>&1 && echo "✅ OK" || echo "❌ FAILED")"
    echo "  NFS mount: $(mountpoint -q /shared && echo "✅ Mounted" || echo "❌ Not mounted")"
    
    echo ""
    echo "MPI Status:"
    echo "  OpenMPI version: $(ompi_info | grep "Open MPI:" | awk '{print $3}')"
    echo "  Hostfile entries: $(wc -l < /shared/hostfile 2>/dev/null || echo "N/A")"
    
    sleep 15
done
EOF

chmod +x /home/ubuntu/monitor_worker.sh
chown ubuntu:ubuntu /home/ubuntu/monitor_worker.sh

# Create worker health check
cat > /home/ubuntu/health_check.sh << 'EOF'
#!/bin/bash

# Health check script for worker node
check_status() {
    local service=$1
    local check_command=$2
    
    if eval $check_command > /dev/null 2>&1; then
        echo "✅ $service: OK"
        return 0
    else
        echo "❌ $service: FAILED"
        return 1
    fi
}

echo "🔍 Worker Node Health Check"
echo "=========================="

# Check system resources
check_status "CPU Load" "[ $(cat /proc/loadavg | awk '{print $1}' | cut -d. -f1) -lt 2 ]"
check_status "Memory" "[ $(free | grep Mem | awk '{print int($3/$2*100)}') -lt 90 ]"
check_status "Disk Space" "[ $(df / | tail -1 | awk '{print int($5)}' | sed 's/%//') -lt 90 ]"

# Check network connectivity
check_status "Master Node Ping" "ping -c 1 -W 3 hpc-master"
check_status "Internet" "ping -c 1 -W 3 8.8.8.8"

# Check services
check_status "SSH Service" "systemctl is-active ssh"
check_status "NFS Mount" "mountpoint -q /shared"

# Check MPI
check_status "MPI Installation" "which mpirun"
check_status "OpenMP Support" "echo | gcc -fopenmp -E -dM - | grep -q _OPENMP"

# Check shared directory access
check_status "Shared Directory Write" "touch /shared/test_worker_${worker_index} && rm /shared/test_worker_${worker_index}"

echo ""
echo "Health check completed at $(date)"
EOF

chmod +x /home/ubuntu/health_check.sh
chown ubuntu:ubuntu /home/ubuntu/health_check.sh

# Set up cron job for periodic health checks
(crontab -u ubuntu -l 2>/dev/null; echo "*/5 * * * * /home/ubuntu/health_check.sh >> /var/log/worker_health.log 2>&1") | crontab -u ubuntu -

# Configure automatic updates
echo "unattended-upgrades unattended-upgrades/enable_auto_updates boolean true" | debconf-set-selections
apt-get install -y unattended-upgrades

# Create worker startup script
cat > /home/ubuntu/worker_startup.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting HPC Worker Node"
echo "=========================="

# Wait for shared directory to be available
while [ ! -d /shared ]; do
    echo "Waiting for shared directory..."
    sleep 5
done

# Ensure proper permissions
chown -R ubuntu:ubuntu /home/ubuntu
chmod 700 /home/ubuntu/.ssh
chmod 600 /home/ubuntu/.ssh/*

# Run health check
/home/ubuntu/health_check.sh

echo "✅ Worker node ready for MPI jobs"
EOF

chmod +x /home/ubuntu/worker_startup.sh
chown ubuntu:ubuntu /home/ubuntu/worker_startup.sh

# Run startup script at boot
echo "@reboot /home/ubuntu/worker_startup.sh >> /var/log/worker_startup.log 2>&1" | crontab -u ubuntu -

# Log completion
echo "Worker node ${worker_index} setup completed at $(date)" >> /var/log/hpc-setup.log

# Run the startup script
sudo -u ubuntu /home/ubuntu/worker_startup.sh
