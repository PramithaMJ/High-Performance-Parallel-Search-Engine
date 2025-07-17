CC=/opt/homebrew/bin/gcc-14
CFLAGS=-Wall -O2 -fopenmp -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp
LDFLAGS=`pkg-config --libs libcurl` -fopenmp -L/opt/homebrew/opt/libomp/lib -lomp
CPPFLAGS=`pkg-config --cflags libcurl` -I/opt/homebrew/opt/libomp/include

# Directories
SRC_DIR=src
INC_DIR=include
OBJ_DIR=obj
BIN_DIR=bin
TEST_DIR=tests

# Source files
BASE_SRC_FILES=$(wildcard $(SRC_DIR)/*.c)
SRC_FILES=$(filter-out $(SRC_DIR)/ranking_enhanced.c, $(BASE_SRC_FILES))
OBJ_FILES=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Enhanced version using ranking_enhanced.c instead of ranking.c
ENHANCED_SRC_FILES=$(filter-out $(SRC_DIR)/ranking.c, $(BASE_SRC_FILES))
ENHANCED_OBJ_FILES=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(ENHANCED_SRC_FILES))

# Header files
INC_FLAGS=-I$(INC_DIR)

# Main targets
all: $(BIN_DIR)/search_engine $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

# Production build - only builds the search engine without tests
production: $(BIN_DIR)/search_engine

# Enhanced version with more OpenMP parallelism
enhanced: $(BIN_DIR)/search_engine_enhanced

# Object file compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INC_FLAGS) -c $< -o $@

# Executable targets
$(BIN_DIR)/search_engine: $(OBJ_FILES)
	$(CC) -o $@ $(filter-out $(OBJ_DIR)/evaluate.o $(OBJ_DIR)/test_%.o, $(OBJ_FILES)) $(LDFLAGS)
	
$(BIN_DIR)/search_engine_enhanced: $(ENHANCED_OBJ_FILES)
	$(CC) -o $@ $(filter-out $(OBJ_DIR)/evaluate.o $(OBJ_DIR)/test_%.o, $(ENHANCED_OBJ_FILES)) $(LDFLAGS)

$(BIN_DIR)/evaluate: $(SRC_DIR)/evaluate.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INC_FLAGS) -o $@ $< $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/evaluate.o, $(OBJ_FILES)) $(LDFLAGS)

$(BIN_DIR)/test_url_normalization: $(TEST_DIR)/test_url_normalization.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -o $@ $<

$(BIN_DIR)/test_medium_urls: $(TEST_DIR)/test_medium_urls.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -o $@ $<

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/search_engine $(BIN_DIR)/search_engine_enhanced $(BIN_DIR)/test_url_normalization $(BIN_DIR)/test_medium_urls $(BIN_DIR)/evaluate

# Make sure the directories exist
$(shell mkdir -p $(SRC_DIR) $(INC_DIR) $(OBJ_DIR) $(BIN_DIR) $(TEST_DIR))
