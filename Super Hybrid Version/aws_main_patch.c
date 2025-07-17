// AWS optimization patch for main.c
// Insert this code at the beginning of main() function after MPI initialization

#ifdef AWS_OPTIMIZED
    // AWS-specific optimizations for t2.medium instances
    
    // Set conservative memory limits
    setenv("OMP_STACKSIZE", "512K", 1);
    setenv("MALLOC_TRIM_THRESHOLD", "100000", 1);
    
    // Limit resources for t2.medium (2 vCPUs, 4GB RAM)
    int aws_max_threads = 2;
    int aws_max_pages = 100;
    int aws_max_depth = 2;
    
    if (mpi_rank == 0) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║                                                          ║\n");
        printf("║         AWS HPC Search Engine Cluster                ║\n");
        printf("║                                                          ║\n");
        printf("║  Instance Type: t2.medium                                ║\n");
        printf("║  MPI Processes: %-3d                                     ║\n", mpi_size);
        printf("║  OpenMP Threads: %-3d                                    ║\n", aws_max_threads);
        printf("║  Total Cores: %-3d                                       ║\n", mpi_size * aws_max_threads);
        printf("║  Memory per Node: 4GB                                   ║\n");
        printf("║  Total Memory: %-3dGB                                    ║\n", mpi_size * 4);
        printf("║                                                          ║\n");
        printf("║  Optimizations: Memory-conscious, Network-tuned         ║\n");
        printf("║                                                          ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n");
        printf("\n");
        
        // Display cluster nodes
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("🌐 Cluster Nodes:\n");
        printf("   • Master: %s (rank 0)\n", hostname);
        for (int i = 1; i < mpi_size; i++) {
            printf("   • Worker %d: hpc-worker-%d (rank %d)\n", i, i, i);
        }
        printf("\n");
    }
    
    // Override default values with AWS-optimized settings
    max_pages = (max_pages > aws_max_pages) ? aws_max_pages : max_pages;
    max_depth = (max_depth > aws_max_depth) ? aws_max_depth : max_depth;
    
    // Ensure thread count doesn't exceed AWS limits
    if (thread_count > aws_max_threads) {
        thread_count = aws_max_threads;
        if (mpi_rank == 0) {
            printf("️  Thread count limited to %d for t2.medium optimization\n", aws_max_threads);
        }
    }
    
    // Set OpenMP environment for AWS
    omp_set_num_threads(thread_count);
    
    // Display AWS-specific configuration
    if (mpi_rank == 0) {
        printf(" AWS Configuration Applied:\n");
        printf("   • Max Pages: %d (memory-optimized)\n", max_pages);
        printf("   • Max Depth: %d (network-optimized)\n", max_depth);
        printf("   • Threads: %d (CPU-optimized)\n", thread_count);
        printf("   • Stack Size: 512KB (memory-optimized)\n");
        printf("\n");
    }
    
    // AWS network timeout settings for crawling
    setenv("CURL_TIMEOUT", "30", 1);
    setenv("CURL_CONNECT_TIMEOUT", "10", 1);
    
#endif // AWS_OPTIMIZED

// Add AWS-specific error handling
#ifdef AWS_OPTIMIZED
void aws_signal_handler(int sig) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf("\n AWS Cluster: Received signal %d, cleaning up...\n", sig);
    }
    
    MPI_Finalize();
    exit(sig);
}

// Register signal handlers for graceful shutdown
signal(SIGINT, aws_signal_handler);
signal(SIGTERM, aws_signal_handler);
#endif
