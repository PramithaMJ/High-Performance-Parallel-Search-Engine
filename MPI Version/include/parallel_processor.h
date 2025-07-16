#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#define PP_BUFFER_SIZE 1048576 // 1MB buffer

// Forward declaration
typedef struct ParallelProcessor ParallelProcessor;

// Callback function type for document processing
typedef void (*ProcessorCallback)(const char* file_path, int file_index, ParallelProcessor* processor);

// Callback function type for token processing
typedef void (*TokenCallback)(const char* token, int token_index, ParallelProcessor* processor);

// Parallel processor
struct ParallelProcessor {
    int mpi_rank;
    int mpi_size;
    int buffer_size;
    char* buffer;
};

// Create a parallel processor
ParallelProcessor* create_parallel_processor(int mpi_rank, int mpi_size);

// Free a parallel processor
void free_parallel_processor(ParallelProcessor* processor);

// Process documents in parallel
void process_documents_parallel(ParallelProcessor* processor, 
                             char file_paths[][256], 
                             int file_count, 
                             ProcessorCallback callback);

// Process batch of tokens in parallel
void process_tokens_parallel(ParallelProcessor* processor, 
                           char** tokens, 
                           int token_count, 
                           TokenCallback callback);

#endif /* PARALLEL_PROCESSOR_H */
