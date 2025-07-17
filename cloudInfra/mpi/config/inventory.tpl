# Ansible Inventory Template
# This file will be populated by Terraform with actual IP addresses

[master]
${master_ip} ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/${ssh_key}

[workers]
%{ for ip in worker_ips ~}
${ip} ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/${ssh_key}
%{ endfor ~}

[mpi_cluster:children]
master
workers

[mpi_cluster:vars]
ansible_ssh_common_args='-o StrictHostKeyChecking=no'
