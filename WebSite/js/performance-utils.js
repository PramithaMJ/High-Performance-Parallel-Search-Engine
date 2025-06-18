// Function to update performance charts with real data
function updatePerformanceCharts(version, metrics) {
    // Get the appropriate chart based on version
    const chartVersionMap = {
        'serial': 'serialPerformanceBreakdown',
        'openmp': 'openmpPerformanceBreakdown',
        'mpi': 'mpiPerformanceBreakdown'
    };
    
    const chartId = chartVersionMap[version];
    const chart = window[chartId];
    
    if (chart) {
        // Update chart data with real metrics
        chart.data.datasets[0].data = [
            metrics.indexing_time_ms || 0,
            metrics.query_time_ms || 0,
            metrics.memory_usage_mb || 0,
            metrics.total_time_ms || 0
        ];
        chart.update();
    }
    
    // Also update the main performance metrics display
    const versionDisplayMap = {
        'serial': 'serialMetrics',
        'openmp': 'openmpMetrics',
        'mpi': 'mpiMetrics'
    };
    
    const metricsContainer = document.getElementById(versionDisplayMap[version]);
    if (metricsContainer) {
        metricsContainer.innerHTML = `
            <p><strong>Query Time:</strong> ${metrics.query_time_ms?.toFixed(2) || 0} ms</p>
            <p><strong>Indexing Time:</strong> ${metrics.indexing_time_ms?.toFixed(2) || 0} ms</p>
            <p><strong>Memory Usage:</strong> ${metrics.memory_usage_mb?.toFixed(2) || 0} MB</p>
            <p><strong>Total Time:</strong> ${metrics.total_time_ms?.toFixed(2) || 0} ms</p>
        `;
    }
}
