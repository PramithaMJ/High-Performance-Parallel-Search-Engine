# MPI Hostfile Template
# This file will be populated by the deployment script with actual IP addresses

# Master node
${master_ip} slots=${slots_per_node}

# Worker nodes
%{ for ip in worker_ips ~}
${ip} slots=${slots_per_node}
%{ endfor ~}
