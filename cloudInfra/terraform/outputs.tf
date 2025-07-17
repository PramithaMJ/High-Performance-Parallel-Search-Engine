# Outputs
output "master_public_ip" {
  description = "Public IP of the master node"
  value       = aws_instance.hpc_master.public_ip
}

output "master_private_ip" {
  description = "Private IP of the master node"
  value       = aws_instance.hpc_master.private_ip
}

output "worker_public_ips" {
  description = "Public IPs of worker nodes"
  value       = aws_instance.hpc_workers[*].public_ip
}

output "worker_private_ips" {
  description = "Private IPs of worker nodes"
  value       = aws_instance.hpc_workers[*].private_ip
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.hpc_alb.dns_name
}

output "ssh_command_master" {
  description = "SSH command to connect to master node"
  value       = "ssh -i ${var.key_name}.pem ubuntu@${aws_instance.hpc_master.public_ip}"
}

output "cluster_info" {
  description = "Cluster information"
  value = {
    cluster_size    = var.cluster_size
    instance_type   = var.instance_type
    total_vcpus     = var.cluster_size * 2  # t2.medium has 2 vCPUs
    total_memory_gb = var.cluster_size * 4  # t2.medium has 4GB RAM
  }
}
