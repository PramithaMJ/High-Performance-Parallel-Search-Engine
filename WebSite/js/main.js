document.addEventListener('DOMContentLoaded', function() {
    // First check that the DOM is fully loaded
    console.log("DOM content loaded, initializing dashboard...");
    
    // Ensure Bootstrap is loaded before initializing tabs
    if (typeof bootstrap === 'undefined') {
        console.error("Bootstrap is not available. Loading fallback...");
        // Create script element for Bootstrap
        const bootstrapScript = document.createElement('script');
        bootstrapScript.src = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js';
        bootstrapScript.onload = function() {
            console.log("Bootstrap loaded dynamically. Initializing tab system...");
            initializeTabSystem();
        };
        document.head.appendChild(bootstrapScript);
    } else {
        // Initialize our custom tab system
        // This fixes the "Uncaught TypeError: Illegal invocation" error in selector-engine.js
        console.log("Bootstrap already loaded. Initializing tab system...");
        initializeTabSystem();
    }
    
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

    // Search button in the search engine tab - using ID selector to avoid jQuery selector issues
    const searchButton = document.getElementById('searchButton');
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            const searchInput = document.querySelector('.search-input');
            const query = searchInput ? searchInput.value : '';
            if (query) {
                executeSearch(query);
            }
        });
    }

    // Also add enter key handling for the search input
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const query = this.value;
                if (query) {
                    executeSearch(query);
                }
            }
        });
    }
    
    // Compare button in the compare tab
    const compareButton = document.querySelector('#compare .btn-primary');
    if (compareButton) {
        compareButton.addEventListener('click', function() {
            // Get selected query set and comparison type
            const querySetSelect = document.querySelector('#compare .form-select:first-of-type');
            const compTypeSelect = document.querySelector('#compare .form-select:nth-of-type(2)');
            
            let querySet = 'default';
            if (querySetSelect) {
                const selectedOption = querySetSelect.options[querySetSelect.selectedIndex].text.toLowerCase();
                if (selectedOption.includes('simple')) {
                    querySet = 'simple';
                } else if (selectedOption.includes('complex')) {
                    querySet = 'complex';
                } else if (selectedOption.includes('custom')) {
                    querySet = 'custom';
                }
            }
            
            let comparisonType = 'performance';
            if (compTypeSelect) {
                const selectedOption = compTypeSelect.options[compTypeSelect.selectedIndex].text.toLowerCase();
                if (selectedOption.includes('memory')) {
                    comparisonType = 'memory';
                } else if (selectedOption.includes('scaling')) {
                    comparisonType = 'scaling';
                } else if (selectedOption.includes('all')) {
                    comparisonType = 'all';
                }
            }
            
            // Run the comparison
            runComparison(querySet, comparisonType);
        });
    }

    // Advanced options toggle
    document.querySelector('.card-header[data-bs-toggle="collapse"]').addEventListener('click', function() {
        this.querySelector('i').classList.toggle('fa-chevron-up');
        this.querySelector('i').classList.toggle('fa-chevron-down');
    });
    
    // Data source selection handler
    const dataSourceSelect = document.getElementById('dataSource');
    if (dataSourceSelect) {
        dataSourceSelect.addEventListener('change', function() {
            const crawlOptions = document.querySelector('.crawl-options');
            if (this.value === 'crawl') {
                crawlOptions.style.display = 'flex';
            } else {
                crawlOptions.style.display = 'none';
            }
        });
    }
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
    // Get the selected engine version safely
    let engineVersion = 'serial'; // Default to serial
    const selectedEngine = document.querySelector('input[name="engineVersion"]:checked');
    if (selectedEngine) {
        engineVersion = selectedEngine.value;
    }
    
    // Get result containers safely
    const resultList = document.querySelector('.result-list');
    const resultCount = document.querySelector('.result-count');
    const resultTime = document.querySelector('.result-time');
    
    // Get advanced options safely with defaults
    const numThreads = document.getElementById('numThreads')?.value || '4';
    const numProcesses = document.getElementById('numProcesses')?.value || '4';
    const resultLimit = document.getElementById('resultLimit')?.value || '10';
    const dataSourceElement = document.getElementById('dataSource');
    const dataSource = dataSourceElement ? dataSourceElement.value : 'dataset';
    const crawlUrl = document.getElementById('crawlUrl')?.value || '';
    const measurePerformance = document.getElementById('measurePerformance')?.checked || false;
    
    // Prepare options
    const options = {
        threads: parseInt(numThreads),
        processes: parseInt(numProcesses),
        limit: parseInt(resultLimit),
        dataSource: dataSource
    };
    
    // Add crawl URL and options if using web crawl
    if (dataSource === 'crawl' && crawlUrl) {
        options.crawlUrl = crawlUrl;
        
        // Add crawl depth and max pages
        const crawlDepth = document.getElementById('crawlDepth');
        const crawlMaxPages = document.getElementById('crawlMaxPages');
        
        if (crawlDepth) {
            options.crawlDepth = parseInt(crawlDepth.value);
        }
        
        if (crawlMaxPages) {
            options.crawlMaxPages = parseInt(crawlMaxPages.value);
        }
    }
    
    // Clear previous results
    resultList.innerHTML = '';
    
    // Show loading
    const loading = document.createElement('div');
    loading.className = 'text-center';
    loading.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Searching...</p>';
    resultList.appendChild(loading);
    
    // Make API call to the search endpoint
    fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            version: engineVersion,
            query: query,
            options: options
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        // Remove loading
        resultList.innerHTML = '';
        
        // Check for errors
        if (data.error) {
            const errorAlert = document.createElement('div');
            errorAlert.className = 'alert alert-danger';
            errorAlert.textContent = `Error: ${data.error}`;
            resultList.appendChild(errorAlert);
            resultCount.textContent = "0 results";
            resultTime.textContent = "in 0.00 ms";
            return;
        }
        
        // Update result info
        const resultCountElement = document.querySelector('.result-count');
        const resultTimeElement = document.querySelector('.result-time');
        const resultsCount = data.results ? data.results.length : 0;
        const execTime = data.execution_time_ms || 0;
        
        if (resultCountElement) {
            resultCountElement.textContent = `${resultsCount} results`;
        }
        
        if (resultTimeElement) {
            resultTimeElement.textContent = `in ${execTime.toFixed(2)} ms`;
        }
        
        // Display results
        if (data.results && data.results.length > 0) {
            data.results.forEach(function(result) {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                resultItem.innerHTML = `
                    <div class="result-title">${result.title || 'Untitled Document'}</div>
                    <div class="result-path">${result.path || 'No path available'}</div>
                    <div class="result-snippet">${result.snippet || 'No snippet available'}</div>
                    <div class="result-meta">
                        <span><strong>Score:</strong> ${(result.score || 0).toFixed(2)}</span>
                    </div>
                `;
                resultList.appendChild(resultItem);
            });
            
            // If we're measuring performance, record it
            if (measurePerformance && data.metrics) {
                updatePerformanceCharts(engineVersion, data.metrics);
            }
        } else {
            const noResults = document.createElement('div');
            noResults.className = 'alert alert-info';
            noResults.textContent = 'No results found for your query.';
            resultList.appendChild(noResults);
        }
    })
    .catch(error => {
        console.error('Error during search:', error);
        resultList.innerHTML = '';
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger';
        errorAlert.textContent = `Search failed: ${error.message}`;
        resultList.appendChild(errorAlert);
        resultCount.textContent = "0 results";
        resultTime.textContent = "in 0.00 ms";
    });
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
