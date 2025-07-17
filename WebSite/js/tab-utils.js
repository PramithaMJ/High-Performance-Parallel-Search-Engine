/**
 * Tab utilities for the parallel search engine dashboard
 * This file contains functions to handle tab navigation and fixes Bootstrap tab issues
 */

// Initialize the tab system
function initializeTabSystem() {
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    tabs.forEach(tab => {
        // Add click handler to each tab
        tab.addEventListener('click', function(event) {
            event.preventDefault();
            
            // Get the target tab pane
            const targetId = this.getAttribute('href');
            const targetPane = document.querySelector(targetId);
            
            if (!targetPane) {
                console.error('Target pane not found:', targetId);
                return;
            }
            
            // Deactivate all tabs and panes
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            
            document.querySelectorAll('[data-bs-toggle="tab"]').forEach(t => {
                t.parentElement.classList.remove('active');
            });
            
            // Activate the target tab and pane
            targetPane.classList.add('active');
            this.parentElement.classList.add('active');
            
            // Update URL hash for bookmarking
            window.location.hash = targetId;
            
            // Custom actions based on tab
            handleTabChange(targetId);
        });
    });
    
    // Check if there's a hash in the URL and activate that tab
    if (window.location.hash) {
        const targetTab = document.querySelector(`[href="${window.location.hash}"]`);
        if (targetTab) {
            targetTab.click();
        }
    }
}

// Function to handle tab change events
function handleTabChange(tabId) {
    // This function can be used to perform custom actions when a tab is selected
    console.log('Tab changed to:', tabId);
    
    switch (tabId) {
        case '#dashboard':
            // Refresh dashboard data
            refreshDashboardData();
            break;
            
        case '#search-engine':
            // Initialize search functionality
            setupSearchInterface();
            break;
            
        case '#performance':
            // Refresh performance charts
            refreshPerformanceCharts();
            break;
            
        case '#compare':
            // Initialize comparison interface
            setupComparisonInterface();
            break;
            
        case '#documentation':
            // Initialize documentation demos and charts
            initializeDocumentation();
            break;
            
        case '#build':
            // Check system status for build interface
            refreshBuildStatus();
            break;
            
        case '#settings':
            // Load saved settings
            loadSavedSettings();
            break;
    }
}

// Function to refresh dashboard data
function refreshDashboardData() {
    console.log('Refreshing dashboard data');
    
    // Fetch the latest metrics
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                // Update charts and stats
                updatePerformanceMetrics(data.metrics);
            }
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
        });
    
    // Check system status
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                // Update version status indicators
                updateVersionStatus(data.versions);
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
}

// Function to setup search interface
function setupSearchInterface() {
    console.log('Setting up search interface');
    
    // Load saved settings
    loadSearchSettings();
}

// Function to refresh performance charts
function refreshPerformanceCharts() {
    console.log('Refreshing performance charts');
    
    // Fetch performance data
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                // Update performance charts
                updatePerformanceHistoryTable(data.metrics);
            }
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
}

// Function to setup comparison interface
function setupComparisonInterface() {
    console.log('Setting up comparison interface');
    
    // No special setup needed yet
}

// Function to refresh build status
function refreshBuildStatus() {
    console.log('Refreshing build status');
    
    // Check system status
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                // Update version status table
                updateVersionStatusTable(data.versions);
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
}

// Function to load saved settings
function loadSavedSettings() {
    console.log('Loading saved settings');
    
    try {
        // Load general settings
        const generalSettings = JSON.parse(localStorage.getItem('generalSettings')) || {};
        
        // Update form values
        if (document.getElementById('default-version')) {
            document.getElementById('default-version').value = generalSettings.defaultVersion || 'openmp';
        }
        
        if (document.getElementById('default-threads')) {
            document.getElementById('default-threads').value = generalSettings.defaultThreads || '4';
        }
        
        if (document.getElementById('default-processes')) {
            document.getElementById('default-processes').value = generalSettings.defaultProcesses || '4';
        }
        
        if (document.getElementById('default-dataset')) {
            document.getElementById('default-dataset').value = generalSettings.defaultDataset || '../dataset';
        }
        
        if (document.getElementById('save-history')) {
            document.getElementById('save-history').checked = generalSettings.saveHistory !== false;
        }
        
        // Load advanced settings
        const advancedSettings = JSON.parse(localStorage.getItem('advancedSettings')) || {};
        
        if (document.getElementById('bm25-k1')) {
            document.getElementById('bm25-k1').value = advancedSettings.bm25k1 || '1.2';
        }
        
        if (document.getElementById('bm25-b')) {
            document.getElementById('bm25-b').value = advancedSettings.bm25b || '0.75';
        }
        
        if (document.getElementById('crawl-timeout')) {
            document.getElementById('crawl-timeout').value = advancedSettings.crawlTimeout || '30';
        }
        
        if (document.getElementById('search-timeout')) {
            document.getElementById('search-timeout').value = advancedSettings.searchTimeout || '10';
        }
        
        if (document.getElementById('use-stemming')) {
            document.getElementById('use-stemming').checked = advancedSettings.useStemming !== false;
        }
        
        if (document.getElementById('remove-stopwords')) {
            document.getElementById('remove-stopwords').checked = advancedSettings.removeStopwords !== false;
        }
        
        // Load MPI settings
        const mpiSettings = JSON.parse(localStorage.getItem('mpiSettings')) || {};
        
        if (document.getElementById('hostfile')) {
            document.getElementById('hostfile').value = mpiSettings.hostfile || '../MPI Version/hostfile';
        }
        
        if (document.getElementById('mpi-options')) {
            document.getElementById('mpi-options').value = mpiSettings.mpiOptions || '';
        }
        
        if (document.getElementById('mpi-debug')) {
            document.getElementById('mpi-debug').checked = mpiSettings.mpiDebug === true;
        }
    } catch (error) {
        console.error('Error loading saved settings:', error);
    }
}
