/**
 * Documentation utilities for the parallel search engine dashboard
 * This file contains functions to handle visual demonstrations and interactive elements
 */

let demoInterval = null;
let demoStartTime = 0;
let demoVersion = '';

// Initialize documentation when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDocumentation();
});

function initializeDocumentation() {
    console.log("Initializing documentation utilities...");
    
    // Setup visual demo grid
    setupProcessingGrid();
    
    // Setup performance charts
    setupDocumentationCharts();
    
    // Setup website crawling options toggle
    setupWebsiteCrawlingToggle();
}

function setupProcessingGrid() {
    const grid = document.getElementById('processing-grid');
    if (!grid) return;
    
    // Create 32 document boxes (4x8 grid)
    grid.innerHTML = '';
    for (let i = 0; i < 32; i++) {
        const docBox = document.createElement('div');
        docBox.className = 'doc-box';
        docBox.id = `doc-${i}`;
        docBox.textContent = `D${i + 1}`;
        grid.appendChild(docBox);
    }
}

function startDemo(version) {
    console.log(`Starting ${version} demo`);
    
    // Reset any existing demo
    resetDemo();
    
    demoVersion = version;
    demoStartTime = Date.now();
    
    // Update description
    updateDemoDescription(version);
    
    // Start the demonstration based on version
    switch(version) {
        case 'serial':
            startSerialDemo();
            break;
        case 'openmp':
            startOpenMPDemo();
            break;
        case 'mpi':
            startMPIDemo();
            break;
        case 'hybrid':
            startHybridDemo();
            break;
    }
}

function resetDemo() {
    if (demoInterval) {
        clearInterval(demoInterval);
        demoInterval = null;
    }
    
    // Reset all document boxes
    const boxes = document.querySelectorAll('.doc-box');
    boxes.forEach(box => {
        box.className = 'doc-box';
    });
    
    // Reset stats
    updateDemoStats(0, 0, 0, 0);
    
    document.getElementById('demo-description').textContent = 'Click a demo button to see how each version processes documents in parallel. Each colored box represents a document being processed.';
}

function updateDemoDescription(version) {
    const descriptions = {
        'serial': 'Serial version processes documents one by one in sequence. Notice how only one document is processed at a time.',
        'openmp': 'OpenMP version uses multiple threads to process documents concurrently. Watch as multiple documents are processed simultaneously.',
        'mpi': 'MPI version distributes documents across multiple processes. Each process works independently on different documents.',
        'hybrid': 'Hybrid version combines MPI processes with OpenMP threads for maximum parallelism. Observe the hierarchical processing pattern.'
    };
    
    document.getElementById('demo-description').textContent = descriptions[version];
}

function startSerialDemo() {
    let currentDoc = 0;
    const totalDocs = 32;
    const processingTime = 200; // ms per document
    
    updateDemoStats(0, 0, 1, 0);
    
    demoInterval = setInterval(() => {
        if (currentDoc < totalDocs) {
            // Process current document
            const box = document.getElementById(`doc-${currentDoc}`);
            box.className = 'doc-box serial-processing processing';
            
            // Complete after processing time
            setTimeout(() => {
                box.className = 'doc-box completed';
                updateDemoStats(Date.now() - demoStartTime, currentDoc + 1, 1, ((currentDoc + 1) / 1) * 100 / totalDocs);
            }, processingTime);
            
            currentDoc++;
        } else {
            clearInterval(demoInterval);
            console.log('Serial demo completed');
        }
    }, processingTime + 50);
}

function startOpenMPDemo() {
    let currentDoc = 0;
    const totalDocs = 32;
    const threads = 4;
    const processingTime = 200; // ms per document
    
    updateDemoStats(0, 0, threads, 0);
    
    demoInterval = setInterval(() => {
        if (currentDoc < totalDocs) {
            // Process up to 'threads' documents concurrently
            for (let t = 0; t < threads && currentDoc < totalDocs; t++) {
                const box = document.getElementById(`doc-${currentDoc}`);
                box.className = 'doc-box openmp-processing processing';
                
                // Complete after processing time
                setTimeout(((docIndex) => {
                    return () => {
                        const completedBox = document.getElementById(`doc-${docIndex}`);
                        completedBox.className = 'doc-box completed';
                        updateDemoStats(Date.now() - demoStartTime, docIndex + 1, threads, ((docIndex + 1) / threads) * 100 / totalDocs);
                    };
                })(currentDoc), processingTime);
                
                currentDoc++;
            }
        } else {
            clearInterval(demoInterval);
            console.log('OpenMP demo completed');
        }
    }, processingTime + 50);
}

function startMPIDemo() {
    let currentDoc = 0;
    const totalDocs = 32;
    const processes = 4;
    const processingTime = 200; // ms per document
    
    updateDemoStats(0, 0, processes, 0);
    
    demoInterval = setInterval(() => {
        if (currentDoc < totalDocs) {
            // Process up to 'processes' documents concurrently
            for (let p = 0; p < processes && currentDoc < totalDocs; p++) {
                const box = document.getElementById(`doc-${currentDoc}`);
                box.className = 'doc-box mpi-processing processing';
                
                // Complete after processing time
                setTimeout(((docIndex) => {
                    return () => {
                        const completedBox = document.getElementById(`doc-${docIndex}`);
                        completedBox.className = 'doc-box completed';
                        updateDemoStats(Date.now() - demoStartTime, docIndex + 1, processes, ((docIndex + 1) / processes) * 100 / totalDocs);
                    };
                })(currentDoc), processingTime);
                
                currentDoc++;
            }
        } else {
            clearInterval(demoInterval);
            console.log('MPI demo completed');
        }
    }, processingTime + 50);
}

function startHybridDemo() {
    let currentDoc = 0;
    const totalDocs = 32;
    const processes = 2;
    const threadsPerProcess = 4;
    const totalWorkers = processes * threadsPerProcess;
    const processingTime = 150; // ms per document (faster due to hybrid efficiency)
    
    updateDemoStats(0, 0, totalWorkers, 0);
    
    demoInterval = setInterval(() => {
        if (currentDoc < totalDocs) {
            // Process up to total workers documents concurrently
            for (let w = 0; w < totalWorkers && currentDoc < totalDocs; w++) {
                const box = document.getElementById(`doc-${currentDoc}`);
                box.className = 'doc-box hybrid-processing processing';
                
                // Complete after processing time
                setTimeout(((docIndex) => {
                    return () => {
                        const completedBox = document.getElementById(`doc-${docIndex}`);
                        completedBox.className = 'doc-box completed';
                        updateDemoStats(Date.now() - demoStartTime, docIndex + 1, totalWorkers, ((docIndex + 1) / totalWorkers) * 100 / totalDocs);
                    };
                })(currentDoc), processingTime);
                
                currentDoc++;
            }
        } else {
            clearInterval(demoInterval);
            console.log('Hybrid demo completed');
        }
    }, processingTime + 30);
}

function updateDemoStats(time, docs, workers, efficiency) {
    document.getElementById('demo-time').textContent = `${time}ms`;
    document.getElementById('demo-docs').textContent = docs;
    document.getElementById('demo-workers').textContent = workers;
    document.getElementById('demo-efficiency').textContent = `${efficiency.toFixed(1)}%`;
}

function setupDocumentationCharts() {
    // Setup speedup chart
    const speedupCtx = document.getElementById('speedupChart');
    if (speedupCtx) {
        new Chart(speedupCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['1', '2', '4', '8', '16'],
                datasets: [
                    {
                        label: 'Ideal Speedup',
                        data: [1, 2, 4, 8, 16],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        borderDash: [5, 5],
                        tension: 0
                    },
                    {
                        label: 'Serial',
                        data: [1, 1, 1, 1, 1],
                        borderColor: '#ff5733',
                        backgroundColor: 'rgba(255, 87, 51, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'OpenMP',
                        data: [1, 1.8, 3.2, 5.5, 8.2],
                        borderColor: '#33ff57',
                        backgroundColor: 'rgba(51, 255, 87, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'MPI',
                        data: [1, 1.9, 3.6, 6.8, 12.1],
                        borderColor: '#3357ff',
                        backgroundColor: 'rgba(51, 87, 255, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Hybrid',
                        data: [1, 1.95, 3.8, 7.2, 14.5],
                        borderColor: '#ae33ff',
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
                            text: 'Number of Processors/Cores'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Parallel Processing Speedup Comparison'
                    }
                }
            }
        });
    }

    // Setup efficiency chart
    const efficiencyCtx = document.getElementById('efficiencyChart');
    if (efficiencyCtx) {
        new Chart(efficiencyCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['1', '2', '4', '8', '16'],
                datasets: [
                    {
                        label: 'Ideal Efficiency',
                        data: [100, 100, 100, 100, 100],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        borderDash: [5, 5],
                        tension: 0
                    },
                    {
                        label: 'Serial',
                        data: [100, 50, 25, 12.5, 6.25],
                        borderColor: '#ff5733',
                        backgroundColor: 'rgba(255, 87, 51, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'OpenMP',
                        data: [100, 90, 80, 69, 51],
                        borderColor: '#33ff57',
                        backgroundColor: 'rgba(51, 255, 87, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'MPI',
                        data: [100, 95, 90, 85, 76],
                        borderColor: '#3357ff',
                        backgroundColor: 'rgba(51, 87, 255, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Hybrid',
                        data: [100, 97.5, 95, 90, 91],
                        borderColor: '#ae33ff',
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
                        max: 100,
                        title: {
                            display: true,
                            text: 'Efficiency (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Number of Processors/Cores'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Parallel Processing Efficiency Comparison'
                    }
                }
            }
        });
    }
}

function setupWebsiteCrawlingToggle() {
    const websiteToggle = document.getElementById('compare-use-website');
    const websiteOptions = document.getElementById('website-crawl-options');
    
    if (websiteToggle && websiteOptions) {
        websiteToggle.addEventListener('change', function() {
            if (this.checked) {
                websiteOptions.style.display = 'block';
            } else {
                websiteOptions.style.display = 'none';
            }
        });
    }
}

// Make functions available globally
window.startDemo = startDemo;
window.resetDemo = resetDemo;
