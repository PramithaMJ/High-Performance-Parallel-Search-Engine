/**
 * Comparison utilities for the parallel search engine dashboard
 * This file contains functions to handle version comparison functionality
 */

// Function to run a comparison between different search engine versions
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
        limit: 10  // Fixed limit for comparisons
    };
    
    // Add website crawling option for comparison
    const useWebsite = document.getElementById('compare-use-website');
    if (useWebsite && useWebsite.checked) {
        const websiteUrl = document.getElementById('compare-website-url').value || "https://medium.com/@lpramithamj";
        options.crawlUrl = websiteUrl;
        options.crawlDepth = parseInt(document.getElementById('compare-website-depth').value || "2");
        options.crawlMaxPages = parseInt(document.getElementById('compare-website-max-pages').value || "10");
        options.dataSource = 'crawl';
    } else {
        // Add dataset options
        if (datasetType === 'custom') {
            const customDataset = prompt('Enter the path to your custom dataset:');
            if (customDataset) {
                options.dataSource = 'custom';
                options.dataPath = customDataset;
            } else {
                options.dataSource = 'dataset'; // Fall back to default
            }
        } else {
            options.dataSource = 'dataset';
        }
    }
    
    // Show loading state
    document.getElementById('run-comparison').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running Comparison...';
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
        document.getElementById('run-comparison').innerHTML = 'Run Comparison';
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
        document.getElementById('run-comparison').innerHTML = 'Run Comparison';
        document.getElementById('run-comparison').disabled = false;
        
        // Show error
        alert('Error connecting to the API: ' + error.message);
        console.error('Error:', error);
    });
}

// Function to update comparison charts
function updateComparisonCharts(results) {
    // Extract data for charts
    const labels = [];
    const queryTimes = [];
    const memoryUsages = [];
    const indexingTimes = [];
    
    // Process results in consistent order
    const versionOrder = ['serial', 'openmp', 'mpi', 'hybrid'];
    
    versionOrder.forEach(version => {
        if (results[version]) {
            labels.push(version.charAt(0).toUpperCase() + version.slice(1));
            queryTimes.push(results[version].metrics?.query_time_ms || 0);
            memoryUsages.push(results[version].metrics?.memory_usage_mb || 0);
            indexingTimes.push(results[version].metrics?.indexing_time_ms || 0);
        }
    });
    
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
    
    // Clear existing rows
    table.innerHTML = '';
    
    // Process results in consistent order
    const versionOrder = ['serial', 'openmp', 'mpi', 'hybrid'];
    
    versionOrder.forEach(version => {
        if (results[version]) {
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
                const maxDisplayLength = 50;
                let title = result.results[0].title;
                if (title.length > maxDisplayLength) {
                    title = title.substring(0, maxDisplayLength) + '...';
                }
                topResultCell.textContent = title;
                topResultCell.setAttribute('title', result.results[0].title); // Full title on hover
            } else {
                topResultCell.textContent = 'No results';
            }
            row.appendChild(topResultCell);
            
            table.appendChild(row);
        }
    });
}

// Function to calculate speedup ratio
function calculateSpeedup(serialTime, parallelTime) {
    if (!serialTime || !parallelTime) return 1;
    return serialTime / parallelTime;
}

// Function to calculate efficiency
function calculateEfficiency(speedup, numProcessors) {
    if (!speedup || !numProcessors) return 0;
    return (speedup / numProcessors) * 100;
}

// Function to calculate overhead
function calculateOverhead(serialTime, parallelTime, numProcessors) {
    if (!serialTime || !parallelTime || !numProcessors) return 0;
    return (numProcessors * parallelTime - serialTime) / serialTime;
}
