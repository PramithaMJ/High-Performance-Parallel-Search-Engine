/**
 * Search utilities for the parallel search engine dashboard
 * This file contains functions to handle search execution, timeouts and retries
 */

// Function to retry a search with a longer timeout
function retryWithLongerTimeout() {
    // Get the current query
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
    const crawlUrl = document.getElementById('crawlUrl')?.value || '';
    
    // Prepare options with extended timeout flag
    const options = {
        threads: parseInt(numThreads),
        processes: parseInt(numProcesses),
        limit: parseInt(resultLimit),
        dataSource: dataSource,
        extendedTimeout: true  // Signal to the API to use an extended timeout
    };
    
    // Add crawl options if applicable
    if (dataSource === 'crawl' && crawlUrl) {
        options.crawlUrl = crawlUrl;
        options.crawlDepth = parseInt(document.getElementById('crawlDepth')?.value || '2');
        options.crawlMaxPages = parseInt(document.getElementById('crawlMaxPages')?.value || '10');
    }
    
    // Show searching again
    const resultList = document.querySelector('.result-list');
    resultList.innerHTML = '';
    
    const loading = document.createElement('div');
    loading.className = 'text-center';
    loading.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Searching with extended timeout (up to 3 minutes)...</p>
    `;
    resultList.appendChild(loading);
    
    // Make API call to search with extended timeout
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
    .then(response => response.json())
    .then(data => {
        // Handle the response the same way as the regular search
        resultList.innerHTML = '';
        
        if (data.error) {
            const errorAlert = document.createElement('div');
            errorAlert.className = 'alert alert-danger';
            errorAlert.innerHTML = `
                <h5>Error: ${data.error}</h5>
                <p>The search could not be completed even with an extended timeout.</p>
                <p>Try running the search monitor tool for detailed diagnostics:</p>
                <pre>python search_monitor.py -v ${engineVersion} -q "${query}" -t 180</pre>
            `;
            resultList.appendChild(errorAlert);
            document.querySelector('.result-count').textContent = "0 results";
            document.querySelector('.result-time').textContent = "in 0.00 ms";
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
        } else {
            const noResults = document.createElement('div');
            noResults.className = 'no-results';
            noResults.textContent = 'No results found';
            resultList.appendChild(noResults);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultList.innerHTML = '';
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger';
        errorAlert.textContent = `Network error: ${error.message}`;
        resultList.appendChild(errorAlert);
    });
}

// Function to optimize the search parameters for better performance
function optimizeSearchParams(query, dataSourceType) {
    // Default optimization settings
    const optimizedParams = {
        threads: 4,
        processes: 4,
        version: 'openmp', // Default to OpenMP for better performance
        limit: 10
    };
    
    // Check query complexity
    const queryWords = query.split(/\s+/).filter(word => word.length > 0);
    const isComplexQuery = queryWords.length > 3 || query.includes('"') || query.length > 30;
    
    // Adjust based on query complexity
    if (isComplexQuery) {
        optimizedParams.threads = 8;
        optimizedParams.processes = 8;
        optimizedParams.version = 'mpi'; // Use MPI for complex queries
    }
    
    // Adjust based on data source
    if (dataSourceType === 'crawl') {
        // Web crawling is expensive, use more resources
        optimizedParams.threads = Math.max(8, optimizedParams.threads);
        optimizedParams.processes = Math.max(8, optimizedParams.processes);
        optimizedParams.version = 'mpi'; // MPI handles web crawling better
        
        // Recommended crawl settings
        optimizedParams.crawlDepth = 2; // Limited depth for faster results
        optimizedParams.crawlMaxPages = 20; // Reasonable number of pages
    }
    
    return optimizedParams;
}
