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
    document.getElementById('search-progress').style.display = 'block';
    document.getElementById('search-progress').innerHTML = `
        <div class="d-flex align-items-center">
            <div class="spinner-border spinner-border-sm me-2" role="status">
                <span class="visually-hidden">Searching...</span>
            </div>
            <div>
                Searching with extended timeout... This may take up to 4 minutes.
            </div>
        </div>
    `;
    document.getElementById('search-error').style.display = 'none';
    
    // API call to search with extended timeout
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
            document.getElementById('search-error').innerHTML = `
                ${data.error || 'An error occurred during the search.'}<br/>
                <small>This may be because the query is too complex or the dataset is too large.</small>
            `;
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
        document.getElementById('search-error').innerHTML = `
            Error connecting to the search API: ${error.message}<br/>
            <small>Check if the server is running and try again.</small>
        `;
        console.error('Error:', error);
    });
}

// Function to handle timeout errors and offer retry
function handleTimeoutError(errorMessage) {
    document.getElementById('search-error').innerHTML = `
        ${errorMessage}<br/>
        <button class="btn btn-warning mt-2" onclick="retryWithLongerTimeout()">
            <i class="fas fa-redo"></i> Retry with Extended Timeout
        </button>
        <small class="d-block mt-1">Note: Extended search may take up to 4 minutes.</small>
    `;
}

// Function to format search results
function formatSearchResults(results) {
    if (!results || results.length === 0) {
        return '<div class="alert alert-info">No results found for your query.</div>';
    }
    
    let html = '';
    
    results.forEach(result => {
        // Format snippet with highlighting if available
        let snippet = result.snippet;
        if (snippet && !snippet.includes('<mark>')) {
            // Simple highlighting of query terms
            const query = document.querySelector('.search-input')?.value || '';
            if (query) {
                const terms = query.split(' ').filter(term => term.length > 2);
                terms.forEach(term => {
                    const regex = new RegExp(`(${term})`, 'gi');
                    snippet = snippet.replace(regex, '<mark>$1</mark>');
                });
            }
        }
        
        html += `
            <div class="result-item">
                <div class="result-title">${result.title}</div>
                <div class="result-path">${result.path}</div>
                <div class="result-snippet">${snippet}</div>
                <div class="result-score">Score: ${result.score.toFixed(2)}</div>
            </div>
        `;
    });
    
    return html;
}

// Function to load saved settings
function loadSearchSettings() {
    // Try to load general settings
    try {
        const generalSettings = JSON.parse(localStorage.getItem('generalSettings'));
        if (generalSettings) {
            // Set default engine version
            const engineVersion = generalSettings.defaultVersion;
            if (engineVersion) {
                const versionRadio = document.querySelector(`input[name="engineVersion"][value="${engineVersion}"]`);
                if (versionRadio) versionRadio.checked = true;
            }
            
            // Set default thread count
            const defaultThreads = generalSettings.defaultThreads;
            if (defaultThreads) {
                const threadsSelect = document.getElementById('numThreads');
                if (threadsSelect) threadsSelect.value = defaultThreads;
            }
            
            // Set default process count
            const defaultProcesses = generalSettings.defaultProcesses;
            if (defaultProcesses) {
                const processesSelect = document.getElementById('numProcesses');
                if (processesSelect) processesSelect.value = defaultProcesses;
            }
        }
    } catch (error) {
        console.error('Error loading search settings:', error);
    }
}

// Initialize search-related functionality when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load saved settings
    loadSearchSettings();
    
    // Setup data source change event
    const dataSource = document.getElementById('dataSource');
    if (dataSource) {
        dataSource.addEventListener('change', function() {
            const value = this.value;
            document.getElementById('customPathOption').style.display = (value === 'custom') ? 'block' : 'none';
            document.getElementById('crawlOptions').style.display = (value === 'crawl') ? 'block' : 'none';
        });
    }
    
    // Setup engine version change event to adjust visible options
    const engineVersionRadios = document.querySelectorAll('input[name="engineVersion"]');
    engineVersionRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            const version = this.value;
            const threadsContainer = document.getElementById('numThreads').closest('.mb-3');
            const processesContainer = document.getElementById('numProcesses').closest('.mb-3');
            
            // Show/hide thread count based on version
            if (version === 'serial') {
                threadsContainer.style.display = 'none';
                processesContainer.style.display = 'none';
            } else if (version === 'openmp') {
                threadsContainer.style.display = 'block';
                processesContainer.style.display = 'none';
            } else if (version === 'mpi') {
                threadsContainer.style.display = 'none';
                processesContainer.style.display = 'block';
            } else if (version === 'hybrid') {
                threadsContainer.style.display = 'block';
                processesContainer.style.display = 'block';
            }
        });
    });
    
    // Trigger change event on initial load
    const checkedRadio = document.querySelector('input[name="engineVersion"]:checked');
    if (checkedRadio) {
        checkedRadio.dispatchEvent(new Event('change'));
    }
});
