# Compiler and flags
CC = mpicc
CFLAGS = -Wall -O2 -D_GNU_SOURCE
CPPFLAGS = `pkg-config --cflags libcurl` -Iinclude
LDFLAGS = `pkg-config --libs libcurl` -lm

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

# List all source files explicitly
SRCS = main.c index.c parser.c crawler.c ranking.c metrics.c load_balancer.c mpi_comm.c dist_index.c parallel_processor.c mpi_helpers.c utils.c evaluate.c benchmark.c
SRC_FILES = $(addprefix $(SRC_DIR)/,$(SRCS))
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Targets
all: $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

production: $(BIN_DIR)/search_engine

# Create directories if missing
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR) $(TEST_DIR))

# Compile .c to .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Link search_engine (exclude test and evaluate objs)
$(BIN_DIR)/search_engine: $(OBJ_FILES)
	$(CC) -o $@ $(filter-out $(OBJ_DIR)/evaluate.o $(OBJ_DIR)/test_%.o, $(OBJ_FILES)) $(LDFLAGS)

# Build evaluate binary
$(BIN_DIR)/evaluate: $(SRC_DIR)/evaluate.c $(OBJ_FILES)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/evaluate.o, $(OBJ_FILES)) $(LDFLAGS)

# Test binaries (now linking with utils.o for shared functions)
$(BIN_DIR)/test_url_normalization: $(TEST_DIR)/test_url_normalization.c $(OBJ_DIR)/utils.o
	$(CC) $(CFLAGS) -I$(INC_DIR) -o $@ $^ $(LDFLAGS)

$(BIN_DIR)/test_medium_urls: $(TEST_DIR)/test_medium_urls.c $(OBJ_DIR)/utils.o
	$(CC) $(CFLAGS) -I$(INC_DIR) -o $@ $^ $(LDFLAGS)

# Optional: Add benchmark binary target
$(BIN_DIR)/benchmark: $(SRC_DIR)/benchmark.c $(OBJ_FILES)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/benchmark.o, $(OBJ_FILES)) $(LDFLAGS)

# Clean command
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate $(BIN_DIR)/benchmark

# Optional: Add MPI run target for testing
run: $(BIN_DIR)/search_engine
	mpirun -np 4 $(BIN_DIR)/search_engine

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build all binaries"
	@echo "  production - Build only search engine"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Run search engine with MPI (4 processes)"
	@echo "  help       - Show this help message"

.PHONY: all production clean run help