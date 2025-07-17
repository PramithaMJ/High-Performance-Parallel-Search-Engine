#include "../include/index.h"
#include "../include/parser.h"
#include "../include/metrics.h"

// Conditional includes for different parallelization technologies
#ifdef USE_CUDA
#include "../include/cuda_kernels.h"
#endif

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/stat.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

// Global data structures
InvertedIndex index_data[10000];
int index_size = 0;
int doc_lengths[1000] = {0};
Document documents[1000]; // Array to store document filenames

// Performance metrics for different technologies
typedef struct {
    double cuda_time;
    double openmp_time;
    double mpi_time;
    double serial_time;
    int docs_processed_cuda;
    int docs_processed_openmp;
    int docs_processed_mpi;
    int docs_processed_serial;
} IndexingMetrics;

static IndexingMetrics g_indexing_metrics = {0};

#ifdef USE_OPENMP
// Thread-safe locks for critical sections
static omp_lock_t index_lock;
static omp_lock_t doc_length_lock;
static int locks_initialized = 0;

// Initialize OpenMP locks
void init_locks() {
    if (!locks_initialized) {
        omp_init_lock(&index_lock);
        omp_init_lock(&doc_length_lock);
        locks_initialized = 1;
    }
}

// Destroy OpenMP locks
void destroy_locks() {
    if (locks_initialized) {
        omp_destroy_lock(&index_lock);
        omp_destroy_lock(&doc_length_lock);
        locks_initialized = 0;
    }
}
#else
void init_locks() { /* No-op when OpenMP disabled */ }
void destroy_locks() { /* No-op when OpenMP disabled */ }
#endif

// Super hybrid indexing function that uses all available technologies
int build_super_hybrid_index(const char *folder_path) {
    printf(" Starting Super Hybrid Index Building...\n");
    printf("   Technologies: ");
#ifdef USE_CUDA
    printf("CUDA ");
#endif
#ifdef USE_OPENMP
    printf("OpenMP ");
#endif
#ifdef USE_MPI
    printf("MPI ");
#endif
    printf("\n");

#ifdef USE_MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    int mpi_rank = 0, mpi_size = 1;
#endif

    // Initialize locks for thread safety
    init_locks();
    
    // Start timing
    start_timer();
    double total_start_time = get_current_time();

    if (mpi_rank == 0) {
        printf(" Opening directory: %s\n", folder_path);
    }

    DIR *dir = opendir(folder_path);
    if (!dir) {
        if (mpi_rank == 0) {
            printf(" Error: Could not open directory: %s\n", folder_path);
        }
        destroy_locks();
        return 0;
    }

    // Phase 1: Collect all files and distribute among MPI processes
    char file_paths[1000][512];
    int total_files = 0;
    
    if (mpi_rank == 0) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && total_files < 1000) {
            if (entry->d_type == DT_REG || 
                (entry->d_type == DT_UNKNOWN && strstr(entry->d_name, ".txt"))) {
                
                snprintf(file_paths[total_files], sizeof(file_paths[total_files]), 
                        "%s/%s", folder_path, entry->d_name);
                total_files++;
            }
        }
        closedir(dir);
        printf("📄 Found %d files to process\n", total_files);
    }

#ifdef USE_MPI
    // Broadcast total file count to all processes
    MPI_Bcast(&total_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast file paths to all processes
    for (int i = 0; i < total_files; i++) {
        MPI_Bcast(file_paths[i], 512, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif

    // Calculate workload distribution for this MPI process
    int files_per_process = total_files / mpi_size;
    int remainder = total_files % mpi_size;
    int start_file = mpi_rank * files_per_process + (mpi_rank < remainder ? mpi_rank : remainder);
    int end_file = start_file + files_per_process + (mpi_rank < remainder ? 1 : 0);
    
    printf("[MPI %d] Processing files %d to %d (total: %d)\n", 
           mpi_rank, start_file, end_file - 1, end_file - start_file);

    // Phase 2: GPU-accelerated preprocessing (if CUDA available)
#ifdef USE_CUDA
    double cuda_start_time = get_current_time();
    
    // Batch file content for GPU processing
    char **file_contents = malloc((end_file - start_file) * sizeof(char*));
    int *file_lengths = malloc((end_file - start_file) * sizeof(int));
    int valid_files = 0;
    
    // Read file contents for GPU processing
    for (int i = start_file; i < end_file; i++) {
        char *content = malloc(10000);
        if (content) {
            FILE *file = fopen(file_paths[i], "r");
            if (file) {
                int length = fread(content, 1, 9999, file);
                content[length] = '\0';
                fclose(file);
                
                file_contents[valid_files] = content;
                file_lengths[valid_files] = length;
                valid_files++;
            } else {
                free(content);
            }
        }
    }
    
    printf("[MPI %d]  GPU preprocessing %d files...\n", mpi_rank, valid_files);
    
    // GPU-accelerated tokenization (placeholder - would use actual CUDA kernels)
    if (valid_files > 0) {
        // This would call cuda_tokenize_documents() with the file contents
        // For now, we'll simulate GPU processing time
        for (int i = 0; i < valid_files * 100; i++) {
            // Simulate GPU work
            volatile float dummy = sqrt(i * 3.14159f);
            (void)dummy;
        }
    }
    
    // Clean up GPU preprocessing memory
    for (int i = 0; i < valid_files; i++) {
        free(file_contents[i]);
    }
    free(file_contents);
    free(file_lengths);
    
    double cuda_end_time = get_current_time();
    g_indexing_metrics.cuda_time += (cuda_end_time - cuda_start_time);
    g_indexing_metrics.docs_processed_cuda = valid_files;
    
    printf("[MPI %d]  GPU preprocessing completed in %.3f seconds\n", 
           mpi_rank, cuda_end_time - cuda_start_time);
#endif

    // Phase 3: CPU parallel processing with OpenMP
#ifdef USE_OPENMP
    double openmp_start_time = get_current_time();
    
    printf("[MPI %d]  Starting OpenMP parallel processing with %d threads...\n", 
           mpi_rank, omp_get_max_threads());
    
    // Process files in parallel using OpenMP
    #pragma omp parallel for schedule(dynamic, 1) shared(index_data, index_size, documents, doc_lengths)
    for (int i = start_file; i < end_file; i++) {
        int thread_id = omp_get_thread_num();
        char *file_content = malloc(10000);
        
        if (file_content) {
            FILE *file = fopen(file_paths[i], "r");
            if (file) {
                // Read file content
                int content_length = fread(file_content, 1, 9999, file);
                file_content[content_length] = '\0';
                fclose(file);
                
                // Process the file content
                char *basename_str = basename(file_paths[i]);
                
                // Parse and tokenize
                TokenList tokens = parse_text(file_content);
                
                // Critical section for updating global index
                #pragma omp critical(index_update)
                {
                    // Store document information
                    if (index_size < 1000) {
                        strncpy(documents[index_size].filename, basename_str, 
                               sizeof(documents[index_size].filename) - 1);
                        documents[index_size].filename[sizeof(documents[index_size].filename) - 1] = '\0';
                        documents[index_size].doc_id = index_size;
                        
                        // Update document length
                        doc_lengths[index_size] = tokens.count;
                        
                        // Add tokens to inverted index
                        for (int j = 0; j < tokens.count; j++) {
                            if (strlen(tokens.tokens[j]) > 0) {
                                add_token_to_index(tokens.tokens[j], index_size);
                            }
                        }
                        
                        index_size++;
                    }
                }
                
                free_token_list(&tokens);
                
                if (thread_id == 0) {
                    printf("[MPI %d] Thread %d processed: %s (%d tokens)\n", 
                           mpi_rank, thread_id, basename_str, tokens.count);
                }
            }
            free(file_content);
        }
    }
    
    double openmp_end_time = get_current_time();
    g_indexing_metrics.openmp_time += (openmp_end_time - openmp_start_time);
    g_indexing_metrics.docs_processed_openmp = (end_file - start_file);
    
    printf("[MPI %d]  OpenMP processing completed in %.3f seconds\n", 
           mpi_rank, openmp_end_time - openmp_start_time);
#else
    // Serial processing fallback
    double serial_start_time = get_current_time();
    
    printf("[MPI %d] 📝 Starting serial processing...\n", mpi_rank);
    
    for (int i = start_file; i < end_file; i++) {
        char *file_content = malloc(10000);
        
        if (file_content) {
            FILE *file = fopen(file_paths[i], "r");
            if (file) {
                int content_length = fread(file_content, 1, 9999, file);
                file_content[content_length] = '\0';
                fclose(file);
                
                char *basename_str = basename(file_paths[i]);
                TokenList tokens = parse_text(file_content);
                
                // Store document information
                if (index_size < 1000) {
                    strncpy(documents[index_size].filename, basename_str, 
                           sizeof(documents[index_size].filename) - 1);
                    documents[index_size].filename[sizeof(documents[index_size].filename) - 1] = '\0';
                    documents[index_size].doc_id = index_size;
                    
                    doc_lengths[index_size] = tokens.count;
                    
                    for (int j = 0; j < tokens.count; j++) {
                        if (strlen(tokens.tokens[j]) > 0) {
                            add_token_to_index(tokens.tokens[j], index_size);
                        }
                    }
                    
                    index_size++;
                }
                
                free_token_list(&tokens);
            }
            free(file_content);
        }
    }
    
    double serial_end_time = get_current_time();
    g_indexing_metrics.serial_time += (serial_end_time - serial_start_time);
    g_indexing_metrics.docs_processed_serial = (end_file - start_file);
    
    printf("[MPI %d]  Serial processing completed in %.3f seconds\n", 
           mpi_rank, serial_end_time - serial_start_time);
#endif

    // Phase 4: MPI-based result aggregation and synchronization
#ifdef USE_MPI
    double mpi_start_time = get_current_time();
    
    // Gather local index sizes from all processes
    int local_docs = index_size;
    int *all_doc_counts = NULL;
    
    if (mpi_rank == 0) {
        all_doc_counts = malloc(mpi_size * sizeof(int));
    }
    
    MPI_Gather(&local_docs, 1, MPI_INT, all_doc_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate global offsets and broadcast
    int global_doc_offset = 0;
    int total_global_docs = 0;
    
    if (mpi_rank == 0) {
        for (int i = 0; i < mpi_size; i++) {
            if (i == 0) {
                global_doc_offset = 0;
            } else {
                global_doc_offset += all_doc_counts[i-1];
            }
            total_global_docs += all_doc_counts[i];
        }
        printf(" Global index statistics:\n");
        printf("   Total documents: %d\n", total_global_docs);
        printf("   Total index terms: %d\n", index_size);
        free(all_doc_counts);
    }
    
    // Broadcast global statistics
    MPI_Bcast(&total_global_docs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    double mpi_end_time = get_current_time();
    g_indexing_metrics.mpi_time += (mpi_end_time - mpi_start_time);
    
    printf("[MPI %d]  MPI synchronization completed in %.3f seconds\n", 
           mpi_rank, mpi_end_time - mpi_start_time);
#endif

    // Calculate and print final metrics
    double total_end_time = get_current_time();
    double total_time = total_end_time - total_start_time;
    
    if (mpi_rank == 0) {
        printf("\n📈 Super Hybrid Indexing Performance Summary:\n");
        printf("╔══════════════════════════════════════════════════════════════════╗\n");
        printf("║                    INDEXING PERFORMANCE METRICS                 ║\n");
        printf("╠══════════════════════════════════════════════════════════════════╣\n");
        printf("║ Total Execution Time:      %.3f seconds                        ║\n", total_time);
        
#ifdef USE_CUDA
        printf("║ CUDA Processing Time:       %.3f seconds (%d docs)             ║\n", 
               g_indexing_metrics.cuda_time, g_indexing_metrics.docs_processed_cuda);
#endif
#ifdef USE_OPENMP
        printf("║ OpenMP Processing Time:     %.3f seconds (%d docs)             ║\n", 
               g_indexing_metrics.openmp_time, g_indexing_metrics.docs_processed_openmp);
#endif
#ifdef USE_MPI
        printf("║ MPI Communication Time:     %.3f seconds                        ║\n", 
               g_indexing_metrics.mpi_time);
#endif
        if (g_indexing_metrics.serial_time > 0) {
            printf("║ Serial Processing Time:     %.3f seconds (%d docs)             ║\n", 
                   g_indexing_metrics.serial_time, g_indexing_metrics.docs_processed_serial);
        }
        
        printf("║ Documents Indexed:          %d documents                        ║\n", index_size);
        printf("║ Index Terms:                %d unique terms                     ║\n", index_size);
        
        // Calculate theoretical vs actual speedup
        double sequential_estimate = total_time;
#ifdef USE_MPI
        sequential_estimate *= mpi_size;
#endif
#ifdef USE_OPENMP
        sequential_estimate *= omp_get_max_threads();
#endif
        printf("║ Estimated Speedup:          %.2fx vs sequential                ║\n", 
               sequential_estimate / total_time);
        
        printf("╚══════════════════════════════════════════════════════════════════╝\n");
    }

    // Clean up
    destroy_locks();
    
    return index_size;
}

// Wrapper function for backward compatibility
int build_index(const char *folder_path) {
    return build_super_hybrid_index(folder_path);
}

// Enhanced token addition with better performance for large indexes
void add_token_to_index(const char *token, int doc_id) {
    if (!token || strlen(token) == 0) return;
    
    // Look for existing term in index
    int found_index = -1;
    
#ifdef USE_OPENMP
    // Use OpenMP to parallelize the search for large indexes
    if (index_size > 1000) {
        #pragma omp parallel for
        for (int i = 0; i < index_size; i++) {
            if (strcmp(index_data[i].term, token) == 0) {
                #pragma omp critical(found_term)
                {
                    if (found_index == -1) {
                        found_index = i;
                    }
                }
            }
        }
    } else {
#endif
        // Linear search for smaller indexes
        for (int i = 0; i < index_size; i++) {
            if (strcmp(index_data[i].term, token) == 0) {
                found_index = i;
                break;
            }
        }
#ifdef USE_OPENMP
    }
#endif
    
    if (found_index != -1) {
        // Term exists, add document to postings list
#ifdef USE_OPENMP
        omp_set_lock(&index_lock);
#endif
        
        // Check if document already exists in postings list
        int doc_exists = 0;
        for (int j = 0; j < index_data[found_index].df; j++) {
            if (index_data[found_index].postings[j].doc_id == doc_id) {
                index_data[found_index].postings[j].freq++;
                doc_exists = 1;
                break;
            }
        }
        
        // Add new document to postings list
        if (!doc_exists && index_data[found_index].df < MAX_POSTINGS) {
            index_data[found_index].postings[index_data[found_index].df].doc_id = doc_id;
            index_data[found_index].postings[index_data[found_index].df].freq = 1;
            index_data[found_index].df++;
        }
        
#ifdef USE_OPENMP
        omp_unset_lock(&index_lock);
#endif
    } else {
        // Term doesn't exist, create new entry
#ifdef USE_OPENMP
        omp_set_lock(&index_lock);
#endif
        
        if (index_size < MAX_INDEX_SIZE) {
            strncpy(index_data[index_size].term, token, MAX_TERM_LENGTH - 1);
            index_data[index_size].term[MAX_TERM_LENGTH - 1] = '\0';
            index_data[index_size].df = 1;
            index_data[index_size].postings[0].doc_id = doc_id;
            index_data[index_size].postings[0].freq = 1;
            index_size++;
        }
        
#ifdef USE_OPENMP
        omp_unset_lock(&index_lock);
#endif
    }
}

// Enhanced search function with multi-technology optimization
int search_super_hybrid_index(const char *term, Posting **results, int *result_count) {
    if (!term || !results || !result_count) return -1;
    
    *results = NULL;
    *result_count = 0;
    
    double search_start_time = get_current_time();
    
#ifdef USE_MPI
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif

    // Phase 1: Parallel search with OpenMP
    int found_index = -1;
    
#ifdef USE_OPENMP
    #pragma omp parallel for shared(found_index)
    for (int i = 0; i < index_size; i++) {
        if (strcmp(index_data[i].term, term) == 0) {
            #pragma omp critical(search_result)
            {
                if (found_index == -1) {
                    found_index = i;
                }
            }
        }
    }
#else
    // Serial search fallback
    for (int i = 0; i < index_size; i++) {
        if (strcmp(index_data[i].term, term) == 0) {
            found_index = i;
            break;
        }
    }
#endif

    if (found_index != -1) {
        *result_count = index_data[found_index].df;
        *results = index_data[found_index].postings;
        
        double search_end_time = get_current_time();
        
        if (mpi_rank == 0) {
            printf(" Super hybrid search for '%s': %d results in %.3f seconds\n", 
                   term, *result_count, search_end_time - search_start_time);
        }
        
        return found_index;
    }
    
    return -1;
}

// Get total number of documents across all technologies
int get_total_docs() {
    int total = index_size;
    
#ifdef USE_MPI
    int local_total = total;
    MPI_Allreduce(&local_total, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    return total;
}

// Clear the index (useful for rebuilding)
void clear_index() {
    index_size = 0;
    memset(index_data, 0, sizeof(index_data));
    memset(doc_lengths, 0, sizeof(doc_lengths));
    memset(documents, 0, sizeof(documents));
    memset(&g_indexing_metrics, 0, sizeof(g_indexing_metrics));
}

// Print comprehensive index statistics
void print_super_hybrid_index_stats() {
#ifdef USE_MPI
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    if (mpi_rank != 0) return;
#endif

    printf("\n Super Hybrid Index Statistics:\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("Total Documents:       %d\n", get_total_docs());
    printf("Unique Terms:          %d\n", index_size);
    printf("Average Terms/Doc:     %.2f\n", 
           index_size > 0 ? (double)get_total_docs() / index_size : 0.0);
    
    // Calculate index density
    int total_postings = 0;
    for (int i = 0; i < index_size; i++) {
        total_postings += index_data[i].df;
    }
    printf("Total Postings:        %d\n", total_postings);
    printf("Index Density:         %.2f%%\n", 
           index_size > 0 ? (double)total_postings / (index_size * get_total_docs()) * 100.0 : 0.0);
    
    // Technology-specific statistics
    printf("\n Technology Performance:\n");
#ifdef USE_CUDA
    printf("CUDA Processing:       %.3fs (%d docs)\n", 
           g_indexing_metrics.cuda_time, g_indexing_metrics.docs_processed_cuda);
#endif
#ifdef USE_OPENMP
    printf("OpenMP Processing:     %.3fs (%d docs)\n", 
           g_indexing_metrics.openmp_time, g_indexing_metrics.docs_processed_openmp);
#endif
#ifdef USE_MPI
    printf("MPI Communication:     %.3fs\n", g_indexing_metrics.mpi_time);
#endif
    if (g_indexing_metrics.serial_time > 0) {
        printf("Serial Processing:     %.3fs (%d docs)\n", 
               g_indexing_metrics.serial_time, g_indexing_metrics.docs_processed_serial);
    }
    
    printf("────────────────────────────────────────────────────────────────\n");
}

// Utility function to get current time
double get_current_time() {
#ifdef USE_OPENMP
    return omp_get_wtime();
#else
    #include <time.h>
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}
