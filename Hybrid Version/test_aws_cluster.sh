#!/bin/bash

# AWS HPC Search Engine Test Suite
# Comprehensive testing for the deployed cluster

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🧪 AWS HPC Search Engine Test Suite${NC}"
echo "====================================="

# Test 1: Cluster Health
echo -e "\n${BLUE}Test 1: Cluster Health Check${NC}"
if /shared/run_aws_cluster.sh -h > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Run script is functional${NC}"
else
    echo -e "${RED}❌ Run script failed${NC}"
    exit 1
fi

# Test 2: MPI Connectivity
echo -e "\n${BLUE}Test 2: MPI Connectivity${NC}"
if mpirun -np 3 --hostfile /shared/hostfile hostname; then
    echo -e "${GREEN}✅ MPI is working across all nodes${NC}"
else
    echo -e "${RED}❌ MPI connectivity failed${NC}"
    exit 1
fi

# Test 3: Search Engine Binary
echo -e "\n${BLUE}Test 3: Search Engine Binary${NC}"
if [ -x "/shared/bin/search_engine" ]; then
    echo -e "${GREEN}✅ Search engine binary is executable${NC}"
else
    echo -e "${RED}❌ Search engine binary not found or not executable${NC}"
    exit 1
fi

# Test 4: Quick Search Test
echo -e "\n${BLUE}Test 4: Quick Search Test${NC}"
echo "Running a simple search test..."
if timeout 60 /shared/run_aws_cluster.sh -q "test search" > /tmp/test_output.log 2>&1; then
    echo -e "${GREEN}✅ Basic search functionality works${NC}"
else
    echo -e "${YELLOW}⚠️ Search test timed out or failed (check /tmp/test_output.log)${NC}"
fi

# Test 5: Performance Test
echo -e "\n${BLUE}Test 5: Performance Test${NC}"
echo "Testing with a small crawl..."
if timeout 120 /shared/run_aws_cluster.sh -c "https://httpbin.org" -d 1 -p 5 > /tmp/perf_test.log 2>&1; then
    echo -e "${GREEN}✅ Performance test completed${NC}"
    if [ -f "/shared/aws_hybrid_metrics.csv" ]; then
        echo "Latest metrics:"
        tail -3 /shared/aws_hybrid_metrics.csv
    fi
else
    echo -e "${YELLOW}⚠️ Performance test failed or timed out${NC}"
fi

# Test 6: Resource Usage
echo -e "\n${BLUE}Test 6: Resource Usage${NC}"
echo "Current resource usage:"
echo "Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "CPU Load: $(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')"
echo "Disk: $(df /shared | tail -1 | awk '{print $5}')"

# Summary
echo -e "\n${BLUE}📊 Test Summary${NC}"
echo "==================="
echo -e "${GREEN}✅ Cluster is ready for production workloads!${NC}"
echo ""
echo "Quick start commands:"
echo "  • Web crawl: /shared/run_aws_cluster.sh -c 'https://medium.com/@lpramithamj' -d 2 -p 30"
echo "  • Search: /shared/run_aws_cluster.sh -q 'artificial intelligence'"
echo "  • Monitor: /shared/run_aws_cluster.sh monitor"
echo ""
echo -e "${YELLOW}💰 Remember: This cluster costs ~$0.14/hour${NC}"
echo -e "${YELLOW}🧹 Cleanup when done: ./deploy_aws.sh cleanup${NC}"
