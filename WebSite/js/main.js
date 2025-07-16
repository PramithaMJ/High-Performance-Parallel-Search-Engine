// main.js - Main JavaScript file for the High-Performance Search Engine Dashboard

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
        console.log("Bootstrap already loaded. Initializing tab system...");
        initializeTabSystem();
    }
    
    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebarCollapse');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            const sidebar = document.getElementById('sidebar');
            const content = document.getElementById('content');
            if (sidebar) sidebar.classList.toggle('active');
            if (content) content.classList.toggle('active');
            console.log("Sidebar toggle clicked, sidebar active:", sidebar.classList.contains('active'));
        });
    } else {
        console.error("Sidebar toggle button not found!");
    }

    // Setup charts
    setupDashboardCharts();
    setupPerformanceCharts();
    setupComparisonCharts();

    // Setup event handlers
    setupEventHandlers();

    // Check system status
    checkSystemStatus();

    // Fetch metrics data
    fetchMetricsData();
});

// Function to initialize tab system
function initializeTabSystem() {
    // Check if the bootstrap-tab-fix.js has already initialized tabs
    if (window.tabsInitializedByFix) {
        console.log("Tabs already initialized by bootstrap-tab-fix.js");
        return;
    }
    
    // Use Bootstrap's tab API to initialize tabs
    const triggerTabList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    triggerTabList.forEach(function(triggerEl) {
        const tabTrigger = new bootstrap.Tab(triggerEl);
        
        triggerEl.addEventListener('click', function(event) {
            event.preventDefault();
            tabTrigger.show();
        });
    });
    
    // Add custom tab navigation fixes
    document.querySelectorAll('.nav-link').forEach(function(navLink) {
        navLink.addEventListener('click', function(event) {
            // Custom navigation code if needed
            console.log("Tab clicked:", this.getAttribute('href'));
        });
    });
}

// Function to setup main dashboard charts
function setupDashboardCharts() {
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    const performanceChart = new Chart(performanceCtx, {
        type: 'bar',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI', 'Hybrid'],
            datasets: [
                {
                    label: 'Average Query Time (ms)',
                    data: [45.3, 20.7, 18.5, 15.2],
                    backgroundColor: [
                        'rgba(255, 87, 51, 0.5)',  // Serial
                        'rgba(51, 255, 87, 0.5)',  // OpenMP
                        'rgba(51, 87, 255, 0.5)',  // MPI
                        'rgba(174, 51, 255, 0.5)'   // Hybrid
                    ],
                    borderColor: [
                        'rgba(255, 87, 51, 1)',
                        'rgba(51, 255, 87, 1)',
                        'rgba(51, 87, 255, 1)',
                        'rgba(174, 51, 255, 1)'
                    ],
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });

    // Memory Chart
    const memoryCtx = document.getElementById('memoryChart').getContext('2d');
    const memoryChart = new Chart(memoryCtx, {
        type: 'bar',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI', 'Hybrid'],
            datasets: [
                {
                    label: 'Memory Usage (MB)',
                    data: [12.4, 14.2, 16.8, 18.5],
                    backgroundColor: [
                        'rgba(255, 87, 51, 0.5)',
                        'rgba(51, 255, 87, 0.5)',
                        'rgba(51, 87, 255, 0.5)',
                        'rgba(174, 51, 255, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 87, 51, 1)',
                        'rgba(51, 255, 87, 1)',
                        'rgba(51, 87, 255, 1)',
                        'rgba(174, 51, 255, 1)'
                    ],
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Memory (MB)'
                    }
                }
            }
        }
    });

    // Resource Chart
    const resourceCtx = document.getElementById('resourceChart').getContext('2d');
    const resourceChart = new Chart(resourceCtx, {
        type: 'line',
        data: {
            labels: ['1', '2', '4', '8', '16'],
            datasets: [
                {
                    label: 'Serial',
                    data: [100, 100, 100, 100, 100],
                    borderColor: 'rgba(255, 87, 51, 1)',
                    backgroundColor: 'rgba(255, 87, 51, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'OpenMP',
                    data: [100, 55, 30, 20, 15],
                    borderColor: 'rgba(51, 255, 87, 1)',
                    backgroundColor: 'rgba(51, 255, 87, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'MPI',
                    data: [100, 60, 35, 25, 20],
                    borderColor: 'rgba(51, 87, 255, 1)',
                    backgroundColor: 'rgba(51, 87, 255, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Hybrid',
                    data: [100, 50, 25, 15, 10],
                    borderColor: 'rgba(174, 51, 255, 1)',
                    backgroundColor: 'rgba(174, 51, 255, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Execution Time (% of Serial)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Number of Threads/Processes'
                    }
                }
            }
        }
    });
}

// Function to setup performance charts
function setupPerformanceCharts() {
    // Query Time Chart
    const queryTimeCtx = document.getElementById('queryTimeChart').getContext('2d');
    const queryTimeChart = new Chart(queryTimeCtx, {
        type: 'line',
        data: {
            labels: ['10KB', '100KB', '1MB', '10MB', '100MB'],
            datasets: [
                {
                    label: 'Serial',
                    data: [10, 50, 200, 800, 3500],
                    borderColor: 'rgba(255, 87, 51, 1)',
                    backgroundColor: 'rgba(255, 87, 51, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'OpenMP',
                    data: [12, 40, 100, 350, 1200],
                    borderColor: 'rgba(51, 255, 87, 1)',
                    backgroundColor: 'rgba(51, 255, 87, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'MPI',
                    data: [15, 45, 120, 300, 900],
                    borderColor: 'rgba(51, 87, 255, 1)',
                    backgroundColor: 'rgba(51, 87, 255, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Hybrid',
                    data: [13, 35, 80, 250, 750],
                    borderColor: 'rgba(174, 51, 255, 1)',
                    backgroundColor: 'rgba(174, 51, 255, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Query Time (ms)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Dataset Size'
                    }
                }
            }
        }
    });

    // Performance Memory Chart
    const performanceMemoryCtx = document.getElementById('performanceMemoryChart').getContext('2d');
    const performanceMemoryChart = new Chart(performanceMemoryCtx, {
        type: 'line',
        data: {
            labels: ['10KB', '100KB', '1MB', '10MB', '100MB'],
            datasets: [
                {
                    label: 'Serial',
                    data: [5, 12, 35, 80, 250],
                    borderColor: 'rgba(255, 87, 51, 1)',
                    backgroundColor: 'rgba(255, 87, 51, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'OpenMP',
                    data: [7, 15, 40, 95, 280],
                    borderColor: 'rgba(51, 255, 87, 1)',
                    backgroundColor: 'rgba(51, 255, 87, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'MPI',
                    data: [8, 18, 50, 120, 320],
                    borderColor: 'rgba(51, 87, 255, 1)',
                    backgroundColor: 'rgba(51, 87, 255, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Hybrid',
                    data: [9, 20, 55, 130, 350],
                    borderColor: 'rgba(174, 51, 255, 1)',
                    backgroundColor: 'rgba(174, 51, 255, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Memory Usage (MB)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Dataset Size'
                    }
                }
            }
        }
    });

    // CPU Utilization Chart
    const cpuUtilizationCtx = document.getElementById('cpuUtilizationChart').getContext('2d');
    const cpuUtilizationChart = new Chart(cpuUtilizationCtx, {
        type: 'line',
        data: {
            labels: ['1', '2', '4', '8', '16'],
            datasets: [
                {
                    label: 'Serial',
                    data: [95, 95, 95, 95, 95],
                    borderColor: 'rgba(255, 87, 51, 1)',
                    backgroundColor: 'rgba(255, 87, 51, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'OpenMP',
                    data: [90, 180, 340, 600, 800],
                    borderColor: 'rgba(51, 255, 87, 1)',
                    backgroundColor: 'rgba(51, 255, 87, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'MPI',
                    data: [95, 190, 360, 650, 850],
                    borderColor: 'rgba(51, 87, 255, 1)',
                    backgroundColor: 'rgba(51, 87, 255, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Hybrid',
                    data: [95, 200, 380, 700, 900],
                    borderColor: 'rgba(174, 51, 255, 1)',
                    backgroundColor: 'rgba(174, 51, 255, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'CPU Utilization (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Number of Threads/Processes'
                    }
                }
            }
        }
    });

    // Indexing Chart
    const indexingCtx = document.getElementById('indexingChart').getContext('2d');
    const indexingChart = new Chart(indexingCtx, {
        type: 'line',
        data: {
            labels: ['10KB', '100KB', '1MB', '10MB', '100MB'],
            datasets: [
                {
                    label: 'Serial',
                    data: [20, 100, 500, 2000, 9000],
                    borderColor: 'rgba(255, 87, 51, 1)',
                    backgroundColor: 'rgba(255, 87, 51, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'OpenMP',
                    data: [25, 80, 300, 1000, 4000],
                    borderColor: 'rgba(51, 255, 87, 1)',
                    backgroundColor: 'rgba(51, 255, 87, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'MPI',
                    data: [30, 90, 350, 1100, 3500],
                    borderColor: 'rgba(51, 87, 255, 1)',
                    backgroundColor: 'rgba(51, 87, 255, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Hybrid',
                    data: [28, 75, 250, 900, 3000],
                    borderColor: 'rgba(174, 51, 255, 1)',
                    backgroundColor: 'rgba(174, 51, 255, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Indexing Time (ms)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Dataset Size'
                    }
                }
            }
        }
    });

    // Scaling Chart
    const scalingCtx = document.getElementById('scalingChart').getContext('2d');
    const scalingChart = new Chart(scalingCtx, {
        type: 'line',
        data: {
            labels: ['1', '2', '4', '8', '16'],
            datasets: [
                {
                    label: 'Ideal Scaling',
                    data: [1, 2, 4, 8, 16],
                    borderColor: 'rgba(0, 0, 0, 0.5)',
                    backgroundColor: 'rgba(0, 0, 0, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.1
                },
                {
                    label: 'OpenMP',
                    data: [1, 1.8, 3.2, 5.1, 7.5],
                    borderColor: 'rgba(51, 255, 87, 1)',
                    backgroundColor: 'rgba(51, 255, 87, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'MPI',
                    data: [1, 1.7, 2.9, 4.8, 7.2],
                    borderColor: 'rgba(51, 87, 255, 1)',
                    backgroundColor: 'rgba(51, 87, 255, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Hybrid',
                    data: [1, 1.9, 3.5, 5.8, 9.1],
                    borderColor: 'rgba(174, 51, 255, 1)',
                    backgroundColor: 'rgba(174, 51, 255, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Speedup Factor'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Number of Threads/Processes'
                    }
                }
            }
        }
    });
}

// Function to setup comparison charts
function setupComparisonCharts() {
    // These charts will be populated when the comparison is run
    // Just set up empty charts for now
    const compareQueryTimeCtx = document.getElementById('compareQueryTimeChart').getContext('2d');
    const compareQueryTimeChart = new Chart(compareQueryTimeCtx, {
        type: 'bar',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI', 'Hybrid'],
            datasets: [
                {
                    label: 'Query Time (ms)',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(255, 87, 51, 0.5)',
                        'rgba(51, 255, 87, 0.5)',
                        'rgba(51, 87, 255, 0.5)',
                        'rgba(174, 51, 255, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 87, 51, 1)',
                        'rgba(51, 255, 87, 1)',
                        'rgba(51, 87, 255, 1)',
                        'rgba(174, 51, 255, 1)'
                    ],
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });

    const compareMemoryCtx = document.getElementById('compareMemoryChart').getContext('2d');
    const compareMemoryChart = new Chart(compareMemoryCtx, {
        type: 'bar',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI', 'Hybrid'],
            datasets: [
                {
                    label: 'Memory Usage (MB)',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(255, 87, 51, 0.5)',
                        'rgba(51, 255, 87, 0.5)',
                        'rgba(51, 87, 255, 0.5)',
                        'rgba(174, 51, 255, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 87, 51, 1)',
                        'rgba(51, 255, 87, 1)',
                        'rgba(51, 87, 255, 1)',
                        'rgba(174, 51, 255, 1)'
                    ],
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Memory (MB)'
                    }
                }
            }
        }
    });

    const compareIndexingCtx = document.getElementById('compareIndexingChart').getContext('2d');
    const compareIndexingChart = new Chart(compareIndexingCtx, {
        type: 'bar',
        data: {
            labels: ['Serial', 'OpenMP', 'MPI', 'Hybrid'],
            datasets: [
                {
                    label: 'Indexing Time (ms)',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(255, 87, 51, 0.5)',
                        'rgba(51, 255, 87, 0.5)',
                        'rgba(51, 87, 255, 0.5)',
                        'rgba(174, 51, 255, 0.5)'
                    ],
                    borderColor: [
                        'rgba(255, 87, 51, 1)',
                        'rgba(51, 255, 87, 1)',
                        'rgba(51, 87, 255, 1)',
                        'rgba(174, 51, 255, 1)'
                    ],
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });
}

// Function to setup event handlers
function setupEventHandlers() {
    // Search button click
    const searchButton = document.getElementById('search-button');
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            performSearch();
        });
    }

    // Search input enter key
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }

    // Data source change
    const dataSource = document.getElementById('dataSource');
    if (dataSource) {
        dataSource.addEventListener('change', function() {
            const value = this.value;
            document.getElementById('customPathOption').style.display = (value === 'custom') ? 'block' : 'none';
            document.getElementById('crawlOptions').style.display = (value === 'crawl') ? 'block' : 'none';
        });
    }

    // Run comparison button
    const runComparisonBtn = document.getElementById('run-comparison');
    if (runComparisonBtn) {
        runComparisonBtn.addEventListener('click', function() {
            runComparison();
        });
    }

    // Build button
    const startBuildBtn = document.getElementById('start-build');
    if (startBuildBtn) {
        startBuildBtn.addEventListener('click', function() {
            startBuild();
        });
    }

    // Settings save buttons
    const saveGeneralSettingsBtn = document.getElementById('save-general-settings');
    if (saveGeneralSettingsBtn) {
        saveGeneralSettingsBtn.addEventListener('click', function() {
            saveGeneralSettings();
        });
    }

    const saveAdvancedSettingsBtn = document.getElementById('save-advanced-settings');
    if (saveAdvancedSettingsBtn) {
        saveAdvancedSettingsBtn.addEventListener('click', function() {
            saveAdvancedSettings();
        });
    }

    const saveMpiSettingsBtn = document.getElementById('save-mpi-settings');
    if (saveMpiSettingsBtn) {
        saveMpiSettingsBtn.addEventListener('click', function() {
            saveMpiSettings();
        });
    }
}

// Function to perform search
function performSearch() {
    const searchInput = document.querySelector('.search-input');
    const query = searchInput ? searchInput.value : '';
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    // Get the selected engine version
    let engineVersion = 'serial'; // Default to serial
    const selectedEngine = document.querySelector('input[name="engineVersion"]:checked');
    if (selectedEngine) {
        engineVersion = selectedEngine.value;
    }
    
    // Get advanced options
    const numThreads = document.getElementById('numThreads')?.value || '4';
    const numProcesses = document.getElementById('numProcesses')?.value || '4';
    const resultLimit = document.getElementById('resultLimit')?.value || '10';
    const dataSourceElement = document.getElementById('dataSource');
    const dataSource = dataSourceElement ? dataSourceElement.value : 'dataset';
    
    // Prepare options
    const options = {
        threads: parseInt(numThreads),
        processes: parseInt(numProcesses),
        limit: parseInt(resultLimit),
        dataSource: dataSource
    };
    
    // Add custom path if applicable
    if (dataSource === 'custom') {
        const customPath = document.getElementById('customPath')?.value;
        if (customPath) {
            options.dataPath = customPath;
        }
    }
    
    // Add crawl options if applicable
    if (dataSource === 'crawl') {
        const crawlUrl = document.getElementById('crawlUrl')?.value;
        if (crawlUrl) {
            options.crawlUrl = crawlUrl;
            options.crawlDepth = parseInt(document.getElementById('crawlDepth')?.value || '2');
            options.crawlMaxPages = parseInt(document.getElementById('crawlMaxPages')?.value || '10');
        } else {
            alert('Please enter a URL to crawl');
            return;
        }
    }
    
    // Show search in progress
    document.getElementById('search-progress').style.display = 'block';
    document.getElementById('search-error').style.display = 'none';
    document.getElementById('search-metrics-panel').style.display = 'none';
    document.getElementById('result-count').style.display = 'none';
    document.querySelector('.result-list').innerHTML = '';
    
    // API call to search
    fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            version: engineVersion,
            options: options
        })
    })
    .then(response => response.json())
    .then(data => {
        // Hide progress indicator
        document.getElementById('search-progress').style.display = 'none';
        
        // Check for errors
        if (data.status === 'error') {
            document.getElementById('search-error').style.display = 'block';
            document.getElementById('search-error').textContent = data.error || 'An error occurred during the search.';
            return;
        }
        
        // Display metrics
        document.getElementById('search-metrics-panel').style.display = 'block';
        document.getElementById('metrics-version').textContent = data.version;
        document.getElementById('metrics-query-time').textContent = data.metrics?.query_time_ms?.toFixed(1) + ' ms';
        document.getElementById('metrics-result-count').textContent = data.result_count;
        document.getElementById('metrics-memory-usage').textContent = data.metrics?.memory_usage_mb?.toFixed(1) + ' MB';
        
        // Display result count
        document.getElementById('result-count').style.display = 'block';
        document.getElementById('result-count-number').textContent = data.result_count;
        
        // Display results
        const resultList = document.querySelector('.result-list');
        resultList.innerHTML = '';
        
        if (data.results && data.results.length > 0) {
            data.results.forEach(result => {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                resultItem.innerHTML = `
                    <div class="result-title">${result.title}</div>
                    <div class="result-path">${result.path}</div>
                    <div class="result-snippet">${result.snippet}</div>
                    <div class="result-score">Score: ${result.score.toFixed(2)}</div>
                `;
                
                resultList.appendChild(resultItem);
            });
        } else {
            resultList.innerHTML = '<div class="alert alert-info">No results found for your query.</div>';
        }
    })
    .catch(error => {
        // Hide progress indicator and show error
        document.getElementById('search-progress').style.display = 'none';
        document.getElementById('search-error').style.display = 'block';
        document.getElementById('search-error').textContent = 'Error connecting to the search API: ' + error.message;
        console.error('Error:', error);
    });
}

// Function to run comparison
function runComparison() {
    const query = document.getElementById('compare-query').value;
    
    if (!query) {
        alert('Please enter a search query for comparison');
        return;
    }
    
    // Get selected versions
    const versions = [];
    if (document.getElementById('compare-serial').checked) versions.push('serial');
    if (document.getElementById('compare-openmp').checked) versions.push('openmp');
    if (document.getElementById('compare-mpi').checked) versions.push('mpi');
    if (document.getElementById('compare-hybrid').checked) versions.push('hybrid');
    
    if (versions.length === 0) {
        alert('Please select at least one version to compare');
        return;
    }
    
    // Get configuration options
    const threads = document.getElementById('compare-threads').value;
    const processes = document.getElementById('compare-processes').value;
    const datasetType = document.getElementById('compare-dataset').value;
    
    // Prepare options
    const options = {
        threads: parseInt(threads),
        processes: parseInt(processes),
        limit: 10,  // Fixed limit for comparisons
        dataSource: 'dataset'  // Default to built-in dataset for fair comparison
    };
    
    // Show loading state
    document.getElementById('run-comparison').textContent = 'Running Comparison...';
    document.getElementById('run-comparison').disabled = true;
    
    // API call to compare
    fetch('/api/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            versions: versions,
            options: options
        })
    })
    .then(response => response.json())
    .then(data => {
        // Reset button state
        document.getElementById('run-comparison').textContent = 'Run Comparison';
        document.getElementById('run-comparison').disabled = false;
        
        // Check for errors
        if (data.status === 'error') {
            alert(data.error || 'An error occurred during comparison.');
            return;
        }
        
        // Show comparison results
        document.getElementById('comparison-results').style.display = 'block';
        
        // Update comparison charts
        updateComparisonCharts(data.results);
        
        // Update comparison table
        updateComparisonTable(data.results);
    })
    .catch(error => {
        // Reset button state
        document.getElementById('run-comparison').textContent = 'Run Comparison';
        document.getElementById('run-comparison').disabled = false;
        
        // Show error
        alert('Error connecting to the API: ' + error.message);
        console.error('Error:', error);
    });
}

// Function to update comparison charts
function updateComparisonCharts(results) {
    // Extract data for charts
    const labels = Object.keys(results).map(version => version.charAt(0).toUpperCase() + version.slice(1));
    const queryTimes = [];
    const memoryUsages = [];
    const indexingTimes = [];
    
    for (const version in results) {
        if (results[version].metrics) {
            queryTimes.push(results[version].metrics.query_time_ms || 0);
            memoryUsages.push(results[version].metrics.memory_usage_mb || 0);
            indexingTimes.push(results[version].metrics.indexing_time_ms || 0);
        } else {
            queryTimes.push(0);
            memoryUsages.push(0);
            indexingTimes.push(0);
        }
    }
    
    // Update Query Time chart
    const queryTimeChart = Chart.getChart('compareQueryTimeChart');
    if (queryTimeChart) {
        queryTimeChart.data.labels = labels;
        queryTimeChart.data.datasets[0].data = queryTimes;
        queryTimeChart.update();
    }
    
    // Update Memory chart
    const memoryChart = Chart.getChart('compareMemoryChart');
    if (memoryChart) {
        memoryChart.data.labels = labels;
        memoryChart.data.datasets[0].data = memoryUsages;
        memoryChart.update();
    }
    
    // Update Indexing chart
    const indexingChart = Chart.getChart('compareIndexingChart');
    if (indexingChart) {
        indexingChart.data.labels = labels;
        indexingChart.data.datasets[0].data = indexingTimes;
        indexingChart.update();
    }
}

// Function to update comparison table
function updateComparisonTable(results) {
    const table = document.getElementById('comparison-table');
    if (!table) return;
    
    table.innerHTML = '';
    
    for (const version in results) {
        const result = results[version];
        const row = document.createElement('tr');
        
        // Version name
        const versionCell = document.createElement('td');
        versionCell.textContent = version.charAt(0).toUpperCase() + version.slice(1);
        row.appendChild(versionCell);
        
        // Query time
        const queryTimeCell = document.createElement('td');
        queryTimeCell.textContent = (result.metrics?.query_time_ms || 0).toFixed(1) + ' ms';
        row.appendChild(queryTimeCell);
        
        // Indexing time
        const indexingTimeCell = document.createElement('td');
        indexingTimeCell.textContent = (result.metrics?.indexing_time_ms || 0).toFixed(1) + ' ms';
        row.appendChild(indexingTimeCell);
        
        // Memory usage
        const memoryUsageCell = document.createElement('td');
        memoryUsageCell.textContent = (result.metrics?.memory_usage_mb || 0).toFixed(1) + ' MB';
        row.appendChild(memoryUsageCell);
        
        // Result count
        const resultCountCell = document.createElement('td');
        resultCountCell.textContent = result.result_count || 0;
        row.appendChild(resultCountCell);
        
        // Top result
        const topResultCell = document.createElement('td');
        if (result.results && result.results.length > 0) {
            topResultCell.textContent = result.results[0].title;
        } else {
            topResultCell.textContent = 'No results';
        }
        row.appendChild(topResultCell);
        
        table.appendChild(row);
    }
}

// Function to start build
function startBuild() {
    // Get build options
    const buildVersionRadio = document.querySelector('input[name="buildVersion"]:checked');
    if (!buildVersionRadio) {
        alert('Please select a version to build');
        return;
    }
    
    const buildVersion = buildVersionRadio.value;
    const cleanBuild = document.getElementById('build-clean').checked;
    const optimize = document.getElementById('build-optimize').checked;
    const debug = document.getElementById('build-debug').checked;
    
    // Show build in progress
    document.getElementById('build-progress').style.display = 'block';
    document.getElementById('build-success').style.display = 'none';
    document.getElementById('build-error').style.display = 'none';
    document.getElementById('build-log-output').textContent = 'Building...';
    document.getElementById('start-build').disabled = true;
    
    // Prepare build options
    const options = {
        clean: cleanBuild,
        optimize: optimize,
        debug: debug
    };
    
    // Handle "all versions" option
    if (buildVersion === 'all') {
        const versions = ['serial', 'openmp', 'mpi', 'hybrid'];
        const buildPromises = versions.map(version => buildSingleVersion(version, options));
        
        Promise.all(buildPromises)
            .then(results => {
                // Combine logs
                const combinedLog = results.map((result, index) => {
                    return `=== Building ${versions[index].toUpperCase()} ===\n${result.log || ''}\n\n`;
                }).join('');
                
                // Check if any build failed
                const anyFailed = results.some(result => !result.success);
                
                if (anyFailed) {
                    document.getElementById('build-progress').style.display = 'none';
                    document.getElementById('build-error').style.display = 'block';
                    document.getElementById('build-error').textContent = 'One or more builds failed. Check the logs for details.';
                } else {
                    document.getElementById('build-progress').style.display = 'none';
                    document.getElementById('build-success').style.display = 'block';
                    document.getElementById('build-success').textContent = 'All builds completed successfully.';
                }
                
                document.getElementById('build-log-output').textContent = combinedLog;
                document.getElementById('start-build').disabled = false;
                
                // Update version status
                checkSystemStatus();
            })
            .catch(error => {
                document.getElementById('build-progress').style.display = 'none';
                document.getElementById('build-error').style.display = 'block';
                document.getElementById('build-error').textContent = 'Build process failed: ' + error.message;
                document.getElementById('start-build').disabled = false;
            });
    } else {
        // Build single version
        buildSingleVersion(buildVersion, options)
            .then(result => {
                if (result.success) {
                    document.getElementById('build-progress').style.display = 'none';
                    document.getElementById('build-success').style.display = 'block';
                    document.getElementById('build-success').textContent = `${buildVersion.toUpperCase()} build completed successfully.`;
                } else {
                    document.getElementById('build-progress').style.display = 'none';
                    document.getElementById('build-error').style.display = 'block';
                    document.getElementById('build-error').textContent = `${buildVersion.toUpperCase()} build failed. Check the logs for details.`;
                }
                
                document.getElementById('build-log-output').textContent = result.log || '';
                document.getElementById('start-build').disabled = false;
                
                // Update version status
                checkSystemStatus();
            })
            .catch(error => {
                document.getElementById('build-progress').style.display = 'none';
                document.getElementById('build-error').style.display = 'block';
                document.getElementById('build-error').textContent = 'Build process failed: ' + error.message;
                document.getElementById('start-build').disabled = false;
            });
    }
}

// Function to build a single version
function buildSingleVersion(version, options) {
    return new Promise((resolve, reject) => {
        fetch('/api/build', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                version: version,
                options: options
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                resolve({
                    success: true,
                    log: data.output
                });
            } else {
                resolve({
                    success: false,
                    log: data.error_output || data.error
                });
            }
        })
        .catch(error => {
            reject(error);
        });
    });
}

// Function to save general settings
function saveGeneralSettings() {
    // Get settings
    const defaultVersion = document.getElementById('default-version').value;
    const defaultThreads = document.getElementById('default-threads').value;
    const defaultProcesses = document.getElementById('default-processes').value;
    const defaultDataset = document.getElementById('default-dataset').value;
    const saveHistory = document.getElementById('save-history').checked;
    
    // Save to local storage
    const settings = {
        defaultVersion,
        defaultThreads,
        defaultProcesses,
        defaultDataset,
        saveHistory
    };
    
    localStorage.setItem('generalSettings', JSON.stringify(settings));
    
    // Show success message
    alert('General settings saved successfully');
}

// Function to save advanced settings
function saveAdvancedSettings() {
    // Get settings
    const bm25k1 = document.getElementById('bm25-k1').value;
    const bm25b = document.getElementById('bm25-b').value;
    const crawlTimeout = document.getElementById('crawl-timeout').value;
    const searchTimeout = document.getElementById('search-timeout').value;
    const useStemming = document.getElementById('use-stemming').checked;
    const removeStopwords = document.getElementById('remove-stopwords').checked;
    
    // Save to local storage
    const settings = {
        bm25k1,
        bm25b,
        crawlTimeout,
        searchTimeout,
        useStemming,
        removeStopwords
    };
    
    localStorage.setItem('advancedSettings', JSON.stringify(settings));
    
    // Show success message
    alert('Advanced settings saved successfully');
}

// Function to save MPI settings
function saveMpiSettings() {
    // Get settings
    const hostfile = document.getElementById('hostfile').value;
    const mpiOptions = document.getElementById('mpi-options').value;
    const mpiDebug = document.getElementById('mpi-debug').checked;
    
    // Save to local storage
    const settings = {
        hostfile,
        mpiOptions,
        mpiDebug
    };
    
    localStorage.setItem('mpiSettings', JSON.stringify(settings));
    
    // Show success message
    alert('MPI settings saved successfully');
}

// Function to check system status
function checkSystemStatus() {
    // API call to get status
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                // Update version status indicators
                updateVersionStatus(data.versions);
            } else {
                console.error('Error getting system status:', data.error);
            }
        })
        .catch(error => {
            console.error('Error connecting to the API:', error);
            document.getElementById('system-status').textContent = 'Offline';
            document.getElementById('system-status').className = 'badge bg-danger';
        });
}

// Function to update version status indicators
function updateVersionStatus(versions) {
    // Serial version
    if (versions.serial && versions.serial.available) {
        document.getElementById('serial-status').innerHTML = '<span class="status-badge available">Available</span>';
    } else {
        document.getElementById('serial-status').innerHTML = '<span class="status-badge unavailable">Unavailable</span>';
    }
    
    // OpenMP version
    if (versions.openmp && versions.openmp.available) {
        document.getElementById('openmp-status').innerHTML = '<span class="status-badge available">Available</span>';
    } else {
        document.getElementById('openmp-status').innerHTML = '<span class="status-badge unavailable">Unavailable</span>';
    }
    
    // MPI version
    if (versions.mpi && versions.mpi.available) {
        document.getElementById('mpi-status').innerHTML = '<span class="status-badge available">Available</span>';
    } else {
        document.getElementById('mpi-status').innerHTML = '<span class="status-badge unavailable">Unavailable</span>';
    }
    
    // Hybrid version
    if (versions.hybrid && versions.hybrid.available) {
        document.getElementById('hybrid-status').innerHTML = '<span class="status-badge available">Available</span>';
    } else {
        document.getElementById('hybrid-status').innerHTML = '<span class="status-badge unavailable">Unavailable</span>';
    }
    
    // Update version status table
    updateVersionStatusTable(versions);
}

// Function to update version status table
function updateVersionStatusTable(versions) {
    const table = document.getElementById('version-status-table');
    if (!table) return;
    
    table.innerHTML = '';
    
    for (const version in versions) {
        const versionInfo = versions[version];
        const row = document.createElement('tr');
        
        // Version name
        const nameCell = document.createElement('td');
        nameCell.textContent = version.charAt(0).toUpperCase() + version.slice(1);
        row.appendChild(nameCell);
        
        // Status
        const statusCell = document.createElement('td');
        if (versionInfo.available) {
            statusCell.innerHTML = '<span class="badge bg-success">Available</span>';
        } else {
            statusCell.innerHTML = '<span class="badge bg-danger">Unavailable</span>';
        }
        row.appendChild(statusCell);
        
        // Path
        const pathCell = document.createElement('td');
        pathCell.textContent = versionInfo.executable || 'N/A';
        row.appendChild(pathCell);
        
        // Last build (placeholder)
        const lastBuildCell = document.createElement('td');
        lastBuildCell.textContent = 'Unknown';
        row.appendChild(lastBuildCell);
        
        // Actions
        const actionsCell = document.createElement('td');
        const rebuildBtn = document.createElement('button');
        rebuildBtn.className = 'btn btn-sm btn-primary me-2';
        rebuildBtn.textContent = 'Rebuild';
        rebuildBtn.addEventListener('click', function() {
            // Set the corresponding radio button
            const radioBtn = document.getElementById(`build-${version}`);
            if (radioBtn) radioBtn.checked = true;
            
            // Switch to build tab
            const buildTab = new bootstrap.Tab(document.querySelector('a[href="#build"]'));
            buildTab.show();
            
            // Slight delay to ensure the tab has switched
            setTimeout(() => {
                startBuild();
            }, 100);
        });
        
        const cleanBtn = document.createElement('button');
        cleanBtn.className = 'btn btn-sm btn-danger';
        cleanBtn.textContent = 'Clean';
        cleanBtn.addEventListener('click', function() {
            if (confirm(`Are you sure you want to clean the ${version} version?`)) {
                // Set the build options
                const radioBtn = document.getElementById(`build-${version}`);
                if (radioBtn) radioBtn.checked = true;
                
                document.getElementById('build-clean').checked = true;
                
                // Switch to build tab
                const buildTab = new bootstrap.Tab(document.querySelector('a[href="#build"]'));
                buildTab.show();
                
                // Slight delay to ensure the tab has switched
                setTimeout(() => {
                    startBuild();
                }, 100);
            }
        });
        
        actionsCell.appendChild(rebuildBtn);
        actionsCell.appendChild(cleanBtn);
        row.appendChild(actionsCell);
        
        table.appendChild(row);
    }
}

// Function to fetch metrics data
function fetchMetricsData() {
    // API call to get metrics
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                // Update performance history table
                updatePerformanceHistory(data.metrics);
            } else {
                console.error('Error getting metrics:', data.error);
            }
        })
        .catch(error => {
            console.error('Error connecting to the API:', error);
        });
}

// Function to update performance history table
function updatePerformanceHistory(metrics) {
    const table = document.getElementById('performance-history');
    if (!table || !metrics || !metrics.runs) return;
    
    table.innerHTML = '';
    
    // Take the last 20 runs
    const runs = metrics.runs.slice(-20);
    
    runs.forEach(run => {
        const row = document.createElement('tr');
        
        // Timestamp
        const timestampCell = document.createElement('td');
        const timestamp = new Date(run.timestamp);
        timestampCell.textContent = timestamp.toLocaleString();
        row.appendChild(timestampCell);
        
        // Version
        const versionCell = document.createElement('td');
        versionCell.textContent = run.version.charAt(0).toUpperCase() + run.version.slice(1);
        row.appendChild(versionCell);
        
        // Query
        const queryCell = document.createElement('td');
        queryCell.textContent = run.query;
        row.appendChild(queryCell);
        
        // Query Time
        const queryTimeCell = document.createElement('td');
        queryTimeCell.textContent = (run.metrics?.query_time_ms || 0).toFixed(1);
        row.appendChild(queryTimeCell);
        
        // Memory Usage
        const memoryUsageCell = document.createElement('td');
        memoryUsageCell.textContent = (run.metrics?.memory_usage_mb || 0).toFixed(1);
        row.appendChild(memoryUsageCell);
        
        // Results
        const resultsCell = document.createElement('td');
        resultsCell.textContent = run.result_count || 0;
        row.appendChild(resultsCell);
        
        table.appendChild(row);
    });
}
