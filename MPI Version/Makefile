CC=mpicc
CFLAGS=-Wall -O2
LDFLAGS=`pkg-config --libs libcurl`
CPPFLAGS=`pkg-config --cflags libcurl`

# Directories
SRC_DIR=src
INC_DIR=include
OBJ_DIR=obj
BIN_DIR=bin
TEST_DIR=tests

# Source files - explicitly list to ensure all components are included
SRC_FILES=$(wildcard $(SRC_DIR)/*.c)
OBJ_FILES=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Ensure all new optimization components are included
SRCS = main.c index.c parser.c crawler.c ranking.c metrics.c load_balancer.c mpi_comm.c dist_index.c parallel_processor.c mpi_helpers.c utils.c evaluate.c benchmark.c

# Header files
INC_FLAGS=-I$(INC_DIR)

# Main targets
all: $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

# Production build - only builds the search engine without tests
production: $(BIN_DIR)/search_engine

# Object file compilation rule
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

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

# Make sure the directories exist
$(shell mkdir -p $(SRC_DIR) $(INC_DIR) $(OBJ_DIR) $(BIN_DIR) $(TEST_DIR))
