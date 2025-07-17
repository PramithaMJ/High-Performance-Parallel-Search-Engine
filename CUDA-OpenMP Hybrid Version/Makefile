# Makefile for CUDA+OpenMP Hybrid Parallel Search Engine
# Supports both CPU-only and GPU-accelerated builds

# Project configuration
PROJECT_NAME = cuda_openmp_search_engine
VERSION = 1.0.0

# Directory structure
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
DATA_DIR = data
SCRIPTS_DIR = scripts

# Compiler and tools
CC = gcc
NVCC = nvcc
AR = ar
RM = rm -f
MKDIR = mkdir -p

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= sm_75  # Default to Turing architecture (RTX 20 series)

# Check for CUDA availability
CUDA_AVAILABLE := $(shell which nvcc > /dev/null 2>&1 && echo 1 || echo 0)

# Compiler flags
CFLAGS = -std=c11 -Wall -Wextra -O3 -march=native -mtune=native
CFLAGS += -fopenmp -ffast-math -funroll-loops
CFLAGS += -I$(INC_DIR) -I$(CUDA_PATH)/include

# CUDA compiler flags
NVCCFLAGS = -std=c++11 -O3 -arch=$(CUDA_ARCH)
NVCCFLAGS += -Xcompiler -fopenmp,-ffast-math,-funroll-loops
NVCCFLAGS += -I$(INC_DIR) -I$(CUDA_PATH)/include
NVCCFLAGS += --use_fast_math --ptxas-options=-v

# Linker flags
LDFLAGS = -fopenmp -lm -lcurl

# CUDA libraries
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand -lcublas -lcusolver

# Conditional CUDA support
ifeq ($(CUDA_AVAILABLE),1)
    CFLAGS += -DUSE_CUDA
    LDFLAGS += $(CUDA_LIBS)
    BUILD_TARGET = cuda
else
    BUILD_TARGET = cpu
endif

# Debug build configuration
DEBUG_CFLAGS = -std=c11 -Wall -Wextra -g -O0 -DDEBUG
DEBUG_CFLAGS += -fopenmp -I$(INC_DIR) -I$(CUDA_PATH)/include
DEBUG_NVCCFLAGS = -std=c++11 -g -G -O0 -arch=$(CUDA_ARCH) -DDEBUG
DEBUG_NVCCFLAGS += -Xcompiler -fopenmp -I$(INC_DIR) -I$(CUDA_PATH)/include

# Profile build configuration
PROFILE_CFLAGS = $(CFLAGS) -pg -DPROFILE
PROFILE_NVCCFLAGS = $(NVCCFLAGS) -lineinfo -DPROFILE

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Object files
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CU_OBJECTS = $(CU_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Conditional object inclusion
ifeq ($(CUDA_AVAILABLE),1)
    ALL_OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)
else
    ALL_OBJECTS = $(C_OBJECTS)
endif

# Main executable
EXECUTABLE = $(BIN_DIR)/$(PROJECT_NAME)

# Test executables
TEST_SOURCES = $(wildcard tests/*.c)
TEST_OBJECTS = $(TEST_SOURCES:tests/%.c=$(OBJ_DIR)/test_%.o)
TEST_EXECUTABLES = $(TEST_SOURCES:tests/%.c=$(BIN_DIR)/test_%)

# Default target
.PHONY: all
all: info directories $(BUILD_TARGET)

# Print build information
.PHONY: info
info:
	@echo "===================================================="
	@echo "  CUDA+OpenMP Hybrid Search Engine Build System"
	@echo "===================================================="
	@echo "Project: $(PROJECT_NAME) v$(VERSION)"
	@echo "CUDA Available: $(CUDA_AVAILABLE)"
	@echo "Target Architecture: $(CUDA_ARCH)"
	@echo "Build Target: $(BUILD_TARGET)"
ifeq ($(CUDA_AVAILABLE),1)
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "CUDA Version: $$($(NVCC) --version | grep release | sed 's/.*release \\([0-9.]\\+\\).*/\\1/')"
endif
	@echo "Compiler: $(CC) $$($(CC) --version | head -1)"
	@echo "OpenMP: $$($(CC) -fopenmp --version 2>/dev/null | head -1 || echo 'Available')"
	@echo "===================================================="

# Create directories
.PHONY: directories
directories:
	@$(MKDIR) $(OBJ_DIR) $(BIN_DIR) $(DATA_DIR)

# CUDA build target
.PHONY: cuda
cuda: $(EXECUTABLE)
ifeq ($(CUDA_AVAILABLE),0)
	@echo "WARNING: CUDA requested but not available, building CPU-only version"
	@$(MAKE) cpu
endif

# CPU-only build target
.PHONY: cpu
cpu: CFLAGS += -DCPU_ONLY
cpu: $(filter-out $(CU_OBJECTS), $(ALL_OBJECTS))
	@echo "Building CPU-only version..."
	$(CC) $(filter-out $(CU_OBJECTS), $(ALL_OBJECTS)) -o $(EXECUTABLE) $(LDFLAGS)
	@echo "CPU-only build complete: $(EXECUTABLE)"

# Debug build
.PHONY: debug
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: NVCCFLAGS = $(DEBUG_NVCCFLAGS)
debug: clean directories $(EXECUTABLE)
	@echo "Debug build complete: $(EXECUTABLE)"

# Profile build
.PHONY: profile
profile: CFLAGS = $(PROFILE_CFLAGS)
profile: NVCCFLAGS = $(PROFILE_NVCCFLAGS)
profile: clean directories $(EXECUTABLE)
	@echo "Profile build complete: $(EXECUTABLE)"

# Production build with optimizations
.PHONY: production
production: CFLAGS += -DNDEBUG -flto
production: NVCCFLAGS += -DNDEBUG
production: LDFLAGS += -flto
production: clean directories $(EXECUTABLE)
	@echo "Production build complete: $(EXECUTABLE)"
	@strip $(EXECUTABLE)

# Main executable
$(EXECUTABLE): $(ALL_OBJECTS)
	@echo "Linking $(EXECUTABLE)..."
	$(CC) $(ALL_OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Compile C source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
ifeq ($(CUDA_AVAILABLE),1)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
	@echo "CUDA not available, skipping $<"
endif

# Test builds
.PHONY: tests
tests: $(TEST_EXECUTABLES)

$(BIN_DIR)/test_%: $(OBJ_DIR)/test_%.o $(filter-out $(OBJ_DIR)/main.o, $(ALL_OBJECTS))
	@echo "Building test $@..."
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/test_%.o: tests/%.c
	@echo "Compiling test $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Documentation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	@if command -v doxygen > /dev/null; then \\
		doxygen Doxyfile; \\
	else \\
		echo "Doxygen not found, skipping documentation generation"; \\
	fi

# Installation
.PHONY: install
install: $(EXECUTABLE)
	@echo "Installing $(PROJECT_NAME)..."
	install -D $(EXECUTABLE) $(DESTDIR)/usr/local/bin/$(PROJECT_NAME)
	install -D README.md $(DESTDIR)/usr/local/share/doc/$(PROJECT_NAME)/README.md
	@echo "Installation complete"

# Package creation
.PHONY: package
package: production
	@echo "Creating package..."
	@tar czf $(PROJECT_NAME)-$(VERSION).tar.gz \\
		$(BIN_DIR) $(DATA_DIR) $(SCRIPTS_DIR) \\
		README.md LICENSE Makefile
	@echo "Package created: $(PROJECT_NAME)-$(VERSION).tar.gz"

# Performance benchmarks
.PHONY: benchmark
benchmark: $(EXECUTABLE)
	@echo "Running performance benchmarks..."
	@$(SCRIPTS_DIR)/benchmark.sh

# Run with sample data
.PHONY: run
run: $(EXECUTABLE)
	@echo "Running with sample configuration..."
	$(EXECUTABLE) --gpu --threads 8 --mode hybrid

# Interactive run
.PHONY: interactive
interactive: $(EXECUTABLE)
	@echo "Starting interactive mode..."
	$(EXECUTABLE) --gpu --verbose

# Quick test run
.PHONY: test-run
test-run: $(EXECUTABLE)
	@echo "Running quick test..."
	$(EXECUTABLE) --gpu --query "test query" --benchmark

# Development helpers
.PHONY: format
format:
	@echo "Formatting source code..."
	@find $(SRC_DIR) $(INC_DIR) -name "*.c" -o -name "*.h" -o -name "*.cu" | \\
		xargs clang-format -i -style=file

.PHONY: lint
lint:
	@echo "Linting source code..."
	@find $(SRC_DIR) -name "*.c" | xargs cppcheck --enable=all --inconclusive

# Cleanup targets
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	$(RM) $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d
	$(RM) $(BIN_DIR)/*
	$(RM) *.gcov *.gcda *.gcno gmon.out

.PHONY: distclean
distclean: clean
	@echo "Deep cleaning..."
	$(RM) -r $(OBJ_DIR) $(BIN_DIR)
	$(RM) $(PROJECT_NAME)-*.tar.gz
	$(RM) -r docs/html docs/latex

# System information
.PHONY: sysinfo
sysinfo:
	@echo "=== System Information ==="
	@echo "OS: $$(uname -s) $$(uname -r)"
	@echo "Architecture: $$(uname -m)"
	@echo "CPU: $$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//')"
	@echo "CPU Cores: $$(nproc)"
	@echo "Memory: $$(free -h | grep '^Mem:' | awk '{print $$2}')"
	@echo "GCC Version: $$($(CC) --version | head -1)"
	@if command -v $(NVCC) > /dev/null; then \\
		echo "NVCC Version: $$($(NVCC) --version | grep release)"; \\
		echo "GPU Information:"; \\
		nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "  No NVIDIA GPU detected"; \\
	else \\
		echo "NVCC: Not available"; \\
	fi

# GPU-specific targets
.PHONY: gpu-info
gpu-info:
ifeq ($(CUDA_AVAILABLE),1)
	@echo "=== GPU Information ==="
	@$(NVCC) --version
	@echo ""
	@nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free --format=csv
else
	@echo "CUDA not available on this system"
endif

# Architecture-specific builds
.PHONY: build-all-archs
build-all-archs:
	@echo "Building for multiple CUDA architectures..."
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_50  # Maxwell
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_60  # Pascal
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_70  # Volta
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_75  # Turing
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_80  # Ampere
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_86  # Ampere RTX 30
	@$(MAKE) clean && $(MAKE) cuda CUDA_ARCH=sm_89  # Ada Lovelace

# Dependency generation
-include $(ALL_OBJECTS:.o=.d)

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.c
	@$(CC) $(CFLAGS) -MM -MT '$(@:.d=.o) $@' $< > $@

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cu
ifeq ($(CUDA_AVAILABLE),1)
	@$(NVCC) $(NVCCFLAGS) -M -MT '$(@:.d=.o) $@' $< > $@
endif

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          - Build the application (default)"
	@echo "  cuda         - Build with CUDA support (if available)"
	@echo "  cpu          - Build CPU-only version"
	@echo "  debug        - Build debug version"
	@echo "  profile      - Build with profiling support"
	@echo "  production   - Build optimized production version"
	@echo "  tests        - Build test executables"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Deep clean including directories"
	@echo "  install      - Install the application"
	@echo "  package      - Create distribution package"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  run          - Run with default configuration"
	@echo "  interactive  - Run in interactive mode"
	@echo "  sysinfo      - Display system information"
	@echo "  gpu-info     - Display GPU information"
	@echo "  format       - Format source code"
	@echo "  lint         - Lint source code"
	@echo "  docs         - Generate documentation"
	@echo "  help         - Show this help message"

# Ensure OBJ_DIR exists for dependency files
$(shell mkdir -p $(OBJ_DIR))

# Mark targets that don't create files
.PHONY: all info directories cuda cpu debug profile production tests docs install \\
        package benchmark run interactive test-run format lint clean distclean \\
        sysinfo gpu-info build-all-archs help
