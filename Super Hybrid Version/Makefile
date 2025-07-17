# Compiler and flags for hybrid MPI+OpenMP version
CC=mpicc
CFLAGS=-Wall -O3
LDFLAGS=`pkg-config --libs libcurl`
CPPFLAGS=`pkg-config --cflags libcurl`

# macOS specific OpenMP flags if needed
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # On macOS, we need to use clang's specific flags for OpenMP support
    CFLAGS += -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
    CPPFLAGS += -I/opt/homebrew/opt/libomp/include
else
    # For Linux and other platforms that support standard OpenMP flags
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

# Directories
SRC_DIR=src
INC_DIR=include
OBJ_DIR=obj
BIN_DIR=bin
TEST_DIR=tests

# Source files
SRC_FILES=$(wildcard $(SRC_DIR)/*.c)
OBJ_FILES=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Header files
INC_FLAGS=-I$(INC_DIR)

# Main targets
all: $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

# Production build - only builds the search engine without tests
production: $(BIN_DIR)/search_engine

# Object file compilation rule with OpenMP and MPI support
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INC_FLAGS) -c $< -o $@

# Executable targets
$(BIN_DIR)/search_engine: $(OBJ_FILES)
	$(CC) -o $@ $(filter-out $(OBJ_DIR)/evaluate.o $(OBJ_DIR)/test_%.o, $(OBJ_FILES)) $(LDFLAGS)

$(BIN_DIR)/evaluate: $(SRC_DIR)/evaluate.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INC_FLAGS) -o $@ $< $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/evaluate.o, $(OBJ_FILES)) $(LDFLAGS)

$(BIN_DIR)/test_url_normalization: $(TEST_DIR)/test_url_normalization.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -o $@ $<

$(BIN_DIR)/test_medium_urls: $(TEST_DIR)/test_medium_urls.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -o $@ $<

# Run with MPI and OpenMP (default 4 processes, 4 threads per process)
run: $(BIN_DIR)/search_engine
	OMP_NUM_THREADS=4 mpirun -np 4 $(BIN_DIR)/search_engine -np 4 -t 4

# Run with MPI and OpenMP with custom configuration
# Usage: make run_custom MPI_PROCS=2 OMP_THREADS=8
run_custom: $(BIN_DIR)/search_engine
	OMP_NUM_THREADS=$(OMP_THREADS) mpirun -np $(MPI_PROCS) $(BIN_DIR)/search_engine -np $(MPI_PROCS) -t $(OMP_THREADS)

# Run with MPI and OpenMP with custom configuration plus additional arguments
# Usage: make run_args MPI_PROCS=2 OMP_THREADS=8 ARGS="-q 'deep learning'"
run_args: $(BIN_DIR)/search_engine
	OMP_NUM_THREADS=$(OMP_THREADS) mpirun -np $(MPI_PROCS) $(BIN_DIR)/search_engine -np $(MPI_PROCS) -t $(OMP_THREADS) $(ARGS)

# Run search engine with a single combined command line argument
# Usage: make search HYBRID="-p 4 -t 8 -q 'machine learning'"
search: $(BIN_DIR)/search_engine
	./scripts/run_hybrid.sh $(HYBRID)

# Run performance benchmarks with varying configurations
benchmark: $(BIN_DIR)/search_engine
	@echo "Running performance benchmarks with different MPI process and OpenMP thread combinations..."
	@bash ./scripts/performance_benchmark.sh

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

# Make sure the directories exist
$(shell mkdir -p $(SRC_DIR) $(INC_DIR) $(OBJ_DIR) $(BIN_DIR) $(TEST_DIR))