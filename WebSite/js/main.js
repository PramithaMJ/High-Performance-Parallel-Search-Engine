document.addEventListener('DOMContentLoaded', function() {
    // Sidebar toggle
    document.getElementById('sidebarCollapse').addEventListener('click', function() {
        document.getElementById('sidebar').classList.toggle('active');
        document.getElementById('content').classList.toggle('active');
    });

    // Setup charts
    setupDashboardCharts();
    setupPerformanceCharts();
    setupComparisonCharts();

    // Setup event handlers
    setupEventHandlers();

    // Simulate search functionality
    simulateSearch();
});

// Function to setup main dashboard charts
function setupDashboardCharts() {
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    const performanceChart = new Chart(performanceCtx, {
        type: 'line',
        data: {
            labels: ['1 Thread', '2 Threads', '4 Threads', '8 Threads', '16 Threads', '32 Threads'],
            datasets: [
                {
                    label: 'Serial Version',
                    data: [365, 365, 365, 365, 365, 365],
                    borderColor: '#36b9cc',
                    backgroundColor: 'rgba(54, 185, 204, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#36b9cc',
                    pointRadius: 3
                },
                {
                    label: 'OpenMP Version',
                    data: [324, 186, 124, 98, 87, 85],
                    borderColor: '#1cc88a',
                    backgroundColor: 'rgba(28, 200, 138, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#1cc88a',
                    pointRadius: 3
                },
                {
                    label: 'MPI Version',
                    data: [250, 142, 78, 42, 28, 21],
                    borderColor: '#f6c23e',
                    backgroundColor: 'rgba(246, 194, 62, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#f6c23e',
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Query Processing Time (ms) vs Threads/Processes'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Threads/Processes'
                    }
                }
            }
        }
    });

    // Memory Chart
    const memoryCtx = document.getElementById('memoryChart').getContext('2d');
    const memoryChart = new Chart(memoryCtx, {
        type: 'doughnut',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI'],
            datasets: [{
                data: [245.6, 283.7, 412.9],
                backgroundColor: [
                    '#36b9cc',
                    '#1cc88a',
                    '#f6c23e'
                ],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                title: {
                    display: true,
                    text: 'Memory Usage (MB)'
                }
            }
        }
    });
}

// Function to setup performance tab charts
function setupPerformanceCharts() {
    // Detailed Performance Chart
    const perfDetailCtx = document.getElementById('performanceDetailChart').getContext('2d');
    const perfDetailChart = new Chart(perfDetailCtx, {
        type: 'bar',
        data: {
            labels: ['Indexing Time', 'Query Time', 'Crawling Time'],
            datasets: [
                {
                    label: 'Serial Version',
                    data: [842.5, 365.2, 5236.9],
                    backgroundColor: 'rgba(54, 185, 204, 0.7)',
                    borderColor: '#36b9cc',
                    borderWidth: 1
                },
                {
                    label: 'OpenMP Version',
                    data: [315.8, 124.6, 1752.3],
                    backgroundColor: 'rgba(28, 200, 138, 0.7)',
                    borderColor: '#1cc88a',
                    borderWidth: 1
                },
                {
                    label: 'MPI Version',
                    data: [213.2, 78.3, 982.5],
                    backgroundColor: 'rgba(246, 194, 62, 0.7)',
                    borderColor: '#f6c23e',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Performance Metrics (ms)'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    },
                    type: 'logarithmic'
                },
                x: {
                    title: {
                        display: true,
                        text: 'Operation'
                    }
                }
            }
        }
    });

    // Query Breakdown Chart
    const queryBreakdownCtx = document.getElementById('queryBreakdownChart').getContext('2d');
    const queryBreakdownChart = new Chart(queryBreakdownCtx, {
        type: 'radar',
        data: {
            labels: ['Parsing', 'Term Search', 'Score Calculation', 'Result Sorting'],
            datasets: [
                {
                    label: 'Serial',
                    data: [45, 190, 110, 20],
                    fill: true,
                    backgroundColor: 'rgba(54, 185, 204, 0.2)',
                    borderColor: '#36b9cc',
                    pointBackgroundColor: '#36b9cc',
                    pointRadius: 3
                },
                {
                    label: 'OpenMP',
                    data: [45, 40, 30, 10],
                    fill: true,
                    backgroundColor: 'rgba(28, 200, 138, 0.2)',
                    borderColor: '#1cc88a',
                    pointBackgroundColor: '#1cc88a',
                    pointRadius: 3
                },
                {
                    label: 'MPI',
                    data: [38, 15, 20, 5],
                    fill: true,
                    backgroundColor: 'rgba(246, 194, 62, 0.2)',
                    borderColor: '#f6c23e',
                    pointBackgroundColor: '#f6c23e',
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Query Processing Time Breakdown (ms)'
                }
            },
            scales: {
                r: {
                    beginAtZero: true
                }
            }
        }
    });

    // Memory Time Chart
    const memoryTimeCtx = document.getElementById('memoryTimeChart').getContext('2d');
    const memoryTimeChart = new Chart(memoryTimeCtx, {
        type: 'line',
        data: {
            labels: ['Start', 'Indexing', 'Query 1', 'Query 2', 'Query 3', 'Query 4', 'Query 5'],
            datasets: [
                {
                    label: 'Serial Version',
                    data: [102.4, 185.7, 212.3, 228.5, 236.8, 240.2, 245.6],
                    borderColor: '#36b9cc',
                    backgroundColor: 'rgba(54, 185, 204, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#36b9cc',
                    pointRadius: 3,
                    fill: true
                },
                {
                    label: 'OpenMP Version',
                    data: [105.2, 208.6, 232.9, 254.3, 268.5, 275.3, 283.7],
                    borderColor: '#1cc88a',
                    backgroundColor: 'rgba(28, 200, 138, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#1cc88a',
                    pointRadius: 3,
                    fill: true
                },
                {
                    label: 'MPI Version',
                    data: [120.8, 280.3, 328.5, 365.2, 387.5, 402.6, 412.9],
                    borderColor: '#f6c23e',
                    backgroundColor: 'rgba(246, 194, 62, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#f6c23e',
                    pointRadius: 3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Memory Usage Over Time (MB)'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Memory (MB)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Processing Stage'
                    }
                }
            }
        }
    });
}

// Function to setup comparison tab charts
function setupComparisonCharts() {
    // Performance Comparison Chart
    const perfCompCtx = document.getElementById('performanceComparisonChart').getContext('2d');
    const perfCompChart = new Chart(perfCompCtx, {
        type: 'bar',
        data: {
            labels: ['Indexing Time', 'Query Time', 'Crawling Time'],
            datasets: [
                {
                    label: 'Serial Version',
                    data: [842.5, 365.2, 5236.9],
                    backgroundColor: 'rgba(54, 185, 204, 0.7)',
                    borderColor: '#36b9cc',
                    borderWidth: 1
                },
                {
                    label: 'OpenMP Version',
                    data: [315.8, 124.6, 1752.3],
                    backgroundColor: 'rgba(28, 200, 138, 0.7)',
                    borderColor: '#1cc88a',
                    borderWidth: 1
                },
                {
                    label: 'MPI Version',
                    data: [213.2, 78.3, 982.5],
                    backgroundColor: 'rgba(246, 194, 62, 0.7)',
                    borderColor: '#f6c23e',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Performance Comparison (ms)'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    },
                    type: 'logarithmic'
                },
                x: {
                    title: {
                        display: true,
                        text: 'Operation'
                    }
                }
            }
        }
    });

    // Scaling Comparison Chart
    const scalingCtx = document.getElementById('scalingComparisonChart').getContext('2d');
    const scalingChart = new Chart(scalingCtx, {
        type: 'line',
        data: {
            labels: ['1', '2', '4', '8', '16', '32'],
            datasets: [
                {
                    label: 'OpenMP (Speedup)',
                    data: [1, 1.96, 3.52, 5.83, 7.24, 7.52],
                    borderColor: '#1cc88a',
                    backgroundColor: 'rgba(28, 200, 138, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#1cc88a',
                    pointRadius: 3,
                    fill: false
                },
                {
                    label: 'MPI (Speedup)',
                    data: [1, 2.12, 4.25, 8.03, 15.45, 27.82],
                    borderColor: '#f6c23e',
                    backgroundColor: 'rgba(246, 194, 62, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#f6c23e',
                    pointRadius: 3,
                    fill: false
                },
                {
                    label: 'Ideal Scaling',
                    data: [1, 2, 4, 8, 16, 32],
                    borderColor: '#e74a3b',
                    backgroundColor: 'rgba(231, 74, 59, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointBackgroundColor: '#e74a3b',
                    pointRadius: 3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Scaling Efficiency (Higher is Better)'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Speedup (x times)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Threads/Processes'
                    }
                }
            }
        }
    });

    // Memory Comparison Chart
    const memoryCompCtx = document.getElementById('memoryComparisonChart').getContext('2d');
    const memoryCompChart = new Chart(memoryCompCtx, {
        type: 'bar',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI'],
            datasets: [{
                label: 'Memory Usage (MB)',
                data: [245.6, 283.7, 412.9],
                backgroundColor: [
                    'rgba(54, 185, 204, 0.7)',
                    'rgba(28, 200, 138, 0.7)',
                    'rgba(246, 194, 62, 0.7)'
                ],
                borderColor: [
                    '#36b9cc',
                    '#1cc88a',
                    '#f6c23e'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Memory Usage Comparison'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Memory (MB)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Version'
                    }
                }
            }
        }
    });
}

// Function to setup event handlers
function setupEventHandlers() {
    // Start Build button
    const startBuildBtn = document.getElementById('startBuild');
    if (startBuildBtn) {
        startBuildBtn.addEventListener('click', function() {
            const buildVersion = document.getElementById('buildVersion').value;
            const buildType = document.getElementById('buildType').value;
            const cleanBuild = document.getElementById('cleanBuild').checked;
            
            // Show build commands in terminal
            const terminalContent = document.querySelector('.terminal-content');
            terminalContent.innerHTML = '';
            
            // Add initial command
            addTerminalLine(terminalContent, `$ cd /Users/pramithajayasooriya/Desktop/Academic/Semester 7/HPC/hpc/HPC Project/High-Performance-Parallel-Search-Engine/${buildVersion} Version`);
            
            // Add clean command if checked
            if (cleanBuild) {
                addTerminalLine(terminalContent, '$ make clean');
                addTerminalLine(terminalContent, 'rm -f obj/*.o bin/search_engine bin/test_url_normalization bin/test_medium_urls bin/evaluate');
            }
            
            // Add build command
            if (buildVersion === 'mpi') {
                addTerminalLine(terminalContent, '$ make');
                addTerminalLine(terminalContent, 'mpicc -Wall -Wextra -g -O3 -I./include -c -o obj/main.o src/main.c');
                addTerminalLine(terminalContent, 'mpicc -Wall -Wextra -g -O3 -I./include -c -o obj/parser.o src/parser.c');
                addTerminalLine(terminalContent, 'mpicc -Wall -Wextra -g -O3 -I./include -c -o obj/index.o src/index.c');
                addTerminalLine(terminalContent, 'mpicc -Wall -Wextra -g -O3 -I./include -c -o obj/ranking.o src/ranking.c');
                addTerminalLine(terminalContent, 'mpicc -Wall -Wextra -g -O3 -I./include -c -o obj/crawler.o src/crawler.c');
                addTerminalLine(terminalContent, 'mpicc -o bin/search_engine obj/main.o obj/parser.o obj/index.o obj/ranking.o obj/crawler.o -lcurl');
                addTerminalLine(terminalContent, 'Build complete! MPI executable created at bin/search_engine');
            } else if (buildVersion === 'openmp') {
                addTerminalLine(terminalContent, '$ make');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -fopenmp -I./include -c -o obj/main.o src/main.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -fopenmp -I./include -c -o obj/parser.o src/parser.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -fopenmp -I./include -c -o obj/index.o src/index.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -fopenmp -I./include -c -o obj/ranking.o src/ranking.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -fopenmp -I./include -c -o obj/crawler.o src/crawler.c');
                addTerminalLine(terminalContent, 'gcc -o bin/search_engine obj/main.o obj/parser.o obj/index.o obj/ranking.o obj/crawler.o -fopenmp -lcurl');
                addTerminalLine(terminalContent, 'Build complete! OpenMP executable created at bin/search_engine');
            } else {
                addTerminalLine(terminalContent, '$ make');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -I./include -c -o obj/main.o src/main.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -I./include -c -o obj/parser.o src/parser.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -I./include -c -o obj/index.o src/index.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -I./include -c -o obj/ranking.o src/ranking.c');
                addTerminalLine(terminalContent, 'gcc -Wall -Wextra -g -O3 -I./include -c -o obj/crawler.o src/crawler.c');
                addTerminalLine(terminalContent, 'gcc -o bin/search_engine obj/main.o obj/parser.o obj/index.o obj/ranking.o obj/crawler.o -lcurl');
                addTerminalLine(terminalContent, 'Build complete! Serial executable created at bin/search_engine');
            }
        });
    }

    // Search button in the search engine tab
    const searchButton = document.querySelector('.search-input + button');
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            const query = document.querySelector('.search-input').value;
            if (query) {
                executeSearch(query);
            }
        });

        // Also add enter key handling for the search input
        document.querySelector('.search-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const query = document.querySelector('.search-input').value;
                if (query) {
                    executeSearch(query);
                }
            }
        });
    }

    // Advanced options toggle
    document.querySelector('.card-header[data-bs-toggle="collapse"]').addEventListener('click', function() {
        this.querySelector('i').classList.toggle('fa-chevron-up');
        this.querySelector('i').classList.toggle('fa-chevron-down');
    });
}

// Function to add a line to the terminal
function addTerminalLine(container, text) {
    const line = document.createElement('div');
    line.className = 'terminal-line';
    line.textContent = text;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

// Function to execute a search
function executeSearch(query) {
    const engineVersion = document.querySelector('input[name="engineVersion"]:checked').value;
    const resultList = document.querySelector('.result-list');
    const resultCount = document.querySelector('.result-count');
    const resultTime = document.querySelector('.result-time');
    
    // Clear previous results
    resultList.innerHTML = '';
    
    // Show loading
    const loading = document.createElement('div');
    loading.className = 'text-center';
    loading.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Searching...</p>';
    resultList.appendChild(loading);
    
    // Simulate search delay
    setTimeout(function() {
        // Remove loading
        resultList.innerHTML = '';
        
        // Get execution time based on version
        let execTime = 0;
        switch(engineVersion) {
            case 'serial':
                execTime = 365.2;
                break;
            case 'openmp':
                execTime = 124.6;
                break;
            case 'mpi':
                execTime = 78.3;
                break;
        }
        
        // Update result info
        resultCount.textContent = "5 results";
        resultTime.textContent = `in ${execTime.toFixed(2)} ms`;
        
        // Sample results
        const results = [
            {
                title: "Distributed Tracing with Zipkin in Microservices",
                path: "dataset/medium_trace_the_path_distributed_tracing_with_zipkin_in_microservices-1__by_pramitha_jayasooriya__medium.txt",
                snippet: "...OpenTracing provides a way to trace requests through microservices architecture. Zipkin is one of the most popular distributed tracing systems that implements the OpenTracing specification...",
                score: 0.95,
                terms: 6
            },
            {
                title: "Circuit Breaker Pattern in Microservices",
                path: "dataset/medium_why_do_we_need_to_use_circuit_bracker_pattern_inside_microservices__by_pramitha_jayasooriya__medium.txt",
                snippet: "...The Circuit Breaker pattern prevents an application from performing operations that are likely to fail, protecting the system from cascading failures...",
                score: 0.87,
                terms: 5
            },
            {
                title: "The Race to 1M Tasks: Benchmarking 1 Million Concurrent Tasks",
                path: "dataset/medium_the_race_to_1m_tasks_benchmarking_1_million_concurrent_tasks__by_pramitha_jayasooriya__medium.txt",
                snippet: "...Benchmarking distributed systems at scale presents unique challenges. This article explores techniques for accurately measuring the performance of systems handling millions of concurrent tasks...",
                score: 0.78,
                terms: 4
            },
            {
                title: "Think Parallel, Compute Faster: ISPC and SPMD",
                path: "dataset/medium_think_parallel_compute_faster_ispc_and_spmd__by_pramitha_jayasooriya__may_2025__medium.txt",
                snippet: "...SPMD (Single Program, Multiple Data) is a parallel programming technique where the same program runs on multiple processors but operates on different data sets...",
                score: 0.72,
                terms: 3
            },
            {
                title: "Between You and the Web: Decoding Forward and Reverse Proxies",
                path: "dataset/medium_between_you_and_the_web_decoding_forward_and_reverse_proxies__by_pramitha_jayasooriya__medium.txt",
                snippet: "...Proxies serve as intermediaries between clients and servers, providing benefits like caching, load balancing, and security. They can be configured for different distributed system architectures...",
                score: 0.65,
                terms: 2
            }
        ];
        
        // Add results to the list
        results.forEach(function(result) {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            resultItem.innerHTML = `
                <div class="result-title">${result.title}</div>
                <div class="result-path">${result.path}</div>
                <div class="result-snippet">${result.snippet}</div>
                <div class="result-meta">
                    <span><strong>Score:</strong> ${result.score.toFixed(2)}</span>
                    <span><strong>Matching Terms:</strong> ${result.terms}</span>
                </div>
            `;
            resultList.appendChild(resultItem);
        });
    }, 1500); // Simulate search delay of 1.5 seconds
}

// Simulate search functionality
function simulateSearch() {
    // This function can be expanded to add realistic search behavior
    // For now, we'll just prepare the UI
    const resultList = document.querySelector('.result-list');
    if (resultList) {
        resultList.innerHTML = '<div class="alert alert-info">Enter a search query and press Search to see results.</div>';
    }
}
