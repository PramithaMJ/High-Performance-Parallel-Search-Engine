// Function to run comparison between search engine versions
function runComparison(querySet, comparisonType) {
    // Get the comparison container
    const performanceTab = document.getElementById('performanceComparison');
    const scalingTab = document.getElementById('scalingComparison');
    const memoryTab = document.getElementById('memoryComparison');
    const accuracyTab = document.getElementById('accuracyComparison');
    
    // Show loading
    const loadingHTML = `
        <div class="text-center my-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Running comparison... This may take a few minutes.</p>
        </div>
    `;
    
    performanceTab.innerHTML = loadingHTML;
    scalingTab.innerHTML = loadingHTML;
    memoryTab.innerHTML = loadingHTML;
    accuracyTab.innerHTML = loadingHTML;
    
    // Make API call to the compare endpoint
    fetch('/api/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            querySet: querySet,
            comparisonType: comparisonType
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        // Update the comparison tabs with the real data
        updateComparisonCharts(data, querySet, comparisonType);
        
        // Create and display detailed results in tables
        displayComparisonTables(data, querySet, comparisonType);
    })
    .catch(error => {
        console.error('Error running comparison:', error);
        const errorHTML = `
            <div class="alert alert-danger my-4" role="alert">
                <h4 class="alert-heading">Comparison Failed</h4>
                <p>There was an error running the comparison: ${error.message}</p>
                <hr>
                <p class="mb-0">Please check that all search engine versions are properly built and available.</p>
            </div>
        `;
        
        // Show error in all tabs
        performanceTab.innerHTML = errorHTML;
        scalingTab.innerHTML = errorHTML;
        memoryTab.innerHTML = errorHTML;
        accuracyTab.innerHTML = errorHTML;
    });
}

// Function to update the comparison charts with real data
function updateComparisonCharts(data, querySet, comparisonType) {
    // Get chart references
    const perfCompChart = Chart.getChart('performanceComparisonChart');
    const scalingChart = Chart.getChart('scalingComparisonChart');
    const memoryCompChart = Chart.getChart('memoryComparisonChart');
    
    // Extract metrics from the data
    const serialMetrics = data.serial?.results?.map(r => r.metrics) || [];
    const openmpMetrics = data.openmp?.results?.map(r => r.metrics) || [];
    const mpiMetrics = data.mpi?.results?.map(r => r.metrics) || [];
    
    // Calculate averages
    const calculateAvg = (metrics, key) => {
        return metrics.reduce((sum, m) => sum + (m[key] || 0), 0) / (metrics.length || 1);
    };
    
    // Performance chart data
    if (perfCompChart) {
        perfCompChart.data.datasets[0].data = [
            calculateAvg(serialMetrics, 'indexing_time_ms'),
            calculateAvg(serialMetrics, 'query_time_ms'),
            data.serial?.avg_time || 0
        ];
        
        perfCompChart.data.datasets[1].data = [
            calculateAvg(openmpMetrics, 'indexing_time_ms'),
            calculateAvg(openmpMetrics, 'query_time_ms'),
            data.openmp?.avg_time || 0
        ];
        
        perfCompChart.data.datasets[2].data = [
            calculateAvg(mpiMetrics, 'indexing_time_ms'),
            calculateAvg(mpiMetrics, 'query_time_ms'),
            data.mpi?.avg_time || 0
        ];
        
        perfCompChart.update();
    }
    
    // Scaling chart data - Use actual speedups from the API response
    if (scalingChart && data.openmp?.speedup && data.mpi?.speedup) {
        // Update with real speedup data if available
        scalingChart.data.datasets[0].data[1] = data.openmp.speedup;
        scalingChart.data.datasets[1].data[1] = data.mpi.speedup;
        scalingChart.update();
    }
    
    // Memory chart data
    if (memoryCompChart) {
        memoryCompChart.data.datasets[0].data = [
            calculateAvg(serialMetrics, 'memory_usage_mb'),
            calculateAvg(openmpMetrics, 'memory_usage_mb'),
            calculateAvg(mpiMetrics, 'memory_usage_mb')
        ];
        memoryCompChart.update();
    }
}

// Function to display detailed comparison tables
function displayComparisonTables(data, querySet, comparisonType) {
    // Get containers for tables
    const performanceTab = document.getElementById('performanceComparison');
    
    // Generate HTML for the performance table
    let tableHTML = `
        <div class="table-responsive mt-4">
            <h5>Comparison Results - ${querySet.charAt(0).toUpperCase() + querySet.slice(1)} Queries</h5>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Query</th>
                        <th>Serial Time (ms)</th>
                        <th>OpenMP Time (ms)</th>
                        <th>MPI Time (ms)</th>
                        <th>OpenMP Speedup</th>
                        <th>MPI Speedup</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Add rows for each query if available
    const serialResults = data.serial?.results || [];
    const openmpResults = data.openmp?.results || [];
    const mpiResults = data.mpi?.results || [];
    
    // Combine results - assume they're in the same order
    const maxLen = Math.max(serialResults.length, openmpResults.length, mpiResults.length);
    
    for (let i = 0; i < maxLen; i++) {
        const serial = serialResults[i] || { query: 'N/A', time_ms: 0 };
        const openmp = openmpResults[i] || { time_ms: 0 };
        const mpi = mpiResults[i] || { time_ms: 0 };
        
        // Calculate speedups for this query
        const openmpSpeedup = serial.time_ms > 0 ? (serial.time_ms / openmp.time_ms).toFixed(2) : 'N/A';
        const mpiSpeedup = serial.time_ms > 0 ? (serial.time_ms / mpi.time_ms).toFixed(2) : 'N/A';
        
        tableHTML += `
            <tr>
                <td>${serial.query || 'Unknown'}</td>
                <td>${serial.time_ms.toFixed(2)}</td>
                <td>${openmp.time_ms.toFixed(2)}</td>
                <td>${mpi.time_ms.toFixed(2)}</td>
                <td>${openmpSpeedup}x</td>
                <td>${mpiSpeedup}x</td>
            </tr>
        `;
    }
    
    // Add summary row
    tableHTML += `
                <tr class="table-primary">
                    <th>Average</th>
                    <td>${data.serial?.avg_time?.toFixed(2) || 'N/A'}</td>
                    <td>${data.openmp?.avg_time?.toFixed(2) || 'N/A'}</td>
                    <td>${data.mpi?.avg_time?.toFixed(2) || 'N/A'}</td>
                    <td>${data.openmp?.speedup?.toFixed(2) || 'N/A'}x</td>
                    <td>${data.mpi?.speedup?.toFixed(2) || 'N/A'}x</td>
                </tr>
            </tbody>
        </table>
    </div>
    `;
    
    // Add the chart
    tableHTML += '<div class="chart-container mt-4" style="position: relative; height:400px;"><canvas id="detailedComparisonChart"></canvas></div>';
    
    // Set the content
    performanceTab.innerHTML = tableHTML;
    
    // Create a new chart for the detailed comparison
    const ctx = document.getElementById('detailedComparisonChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: serialResults.map(r => r.query || 'Unknown'),
            datasets: [
                {
                    label: 'Serial Version',
                    data: serialResults.map(r => r.time_ms),
                    backgroundColor: 'rgba(54, 185, 204, 0.7)',
                    borderColor: '#36b9cc',
                    borderWidth: 1
                },
                {
                    label: 'OpenMP Version',
                    data: openmpResults.map(r => r.time_ms),
                    backgroundColor: 'rgba(28, 200, 138, 0.7)',
                    borderColor: '#1cc88a',
                    borderWidth: 1
                },
                {
                    label: 'MPI Version',
                    data: mpiResults.map(r => r.time_ms),
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
                    text: 'Query Processing Time by Version (ms)'
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
                        text: 'Query'
                    }
                }
            }
        }
    });
}
