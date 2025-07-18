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
    setupEnhancedProcessingGrid();
    
    // Setup interactive controls
    setupInteractiveControls();
    
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

// Enhanced grid setup with proper demo grid
function setupEnhancedProcessingGrid() {
    const grid = document.getElementById('demo-grid');
    if (!grid) {
        console.warn("Demo grid not found");
        return;
    }
    
    // Create 32 document boxes (4x8 grid)
    grid.innerHTML = '';
    for (let i = 0; i < 32; i++) {
        const docBox = document.createElement('div');
        docBox.className = 'demo-doc';
        docBox.id = `demo-doc-${i}`;
        docBox.textContent = `Doc ${i + 1}`;
        docBox.setAttribute('data-doc-id', i);
        grid.appendChild(docBox);
    }
    
    console.log("Demo grid setup complete with 32 documents");
}

function startDemo(version) {
    console.log(`Starting ${version} demo`);
    
    // Reset any existing demo
    resetDemo();
    
    demoVersion = version;
    demoStartTime = Date.now();
    
    const docs = document.querySelectorAll('.doc-box');
    let processedCount = 0;
    
    // Update demo stats
    updateDemoStats(version, 0, docs.length, 0);
    
    switch (version) {
        case 'serial':
            runSerialDemo(docs);
            break;
        case 'openmp':
            runOpenMPDemo(docs);
            break;
        case 'mpi':
            runMPIDemo(docs);
            break;
        case 'hybrid':
            runHybridDemo(docs);
            break;
    }
}

// Serial processing demonstration
function runSerialDemo(docs) {
    let index = 0;
    const processTime = 400; // ms per document
    
    demoInterval = setInterval(() => {
        if (index >= docs.length) {
            clearInterval(demoInterval);
            updateFinalStats('serial');
            return;
        }
        
        // Process one document at a time
        const doc = docs[index];
        doc.classList.add('serial-processing', 'processing');
        
        setTimeout(() => {
            doc.classList.remove('serial-processing', 'processing');
            doc.classList.add('completed');
            updateDemoStats('serial', index + 1, docs.length, Date.now() - demoStartTime);
        }, processTime);
        
        index++;
    }, processTime + 50);
}

// OpenMP processing demonstration
function runOpenMPDemo(docs) {
    const threads = 4;
    const processTime = 300; // ms per document
    let processedCount = 0;
    
    // Process in batches of 4 (threads)
    for (let batch = 0; batch < Math.ceil(docs.length / threads); batch++) {
        setTimeout(() => {
            const startIndex = batch * threads;
            const endIndex = Math.min(startIndex + threads, docs.length);
            
            // Process all documents in this batch simultaneously
            for (let i = startIndex; i < endIndex; i++) {
                const doc = docs[i];
                doc.classList.add('openmp-processing', 'processing');
                
                setTimeout(() => {
                    doc.classList.remove('openmp-processing', 'processing');
                    doc.classList.add('completed');
                    processedCount++;
                    updateDemoStats('openmp', processedCount, docs.length, Date.now() - demoStartTime);
                    
                    if (processedCount >= docs.length) {
                        updateFinalStats('openmp');
                    }
                }, processTime);
            }
        }, batch * (processTime + 100));
    }
}

// MPI processing demonstration
function runMPIDemo(docs) {
    const processes = 4;
    const processTime = 250; // ms per document
    let processedCount = 0;
    
    // Distribute documents across processes
    const docsPerProcess = Math.ceil(docs.length / processes);
    
    for (let proc = 0; proc < processes; proc++) {
        const startIndex = proc * docsPerProcess;
        const endIndex = Math.min(startIndex + docsPerProcess, docs.length);
        
        // Each process handles its documents sequentially
        for (let i = startIndex; i < endIndex; i++) {
            const docIndex = i;
            const delay = (i - startIndex) * processTime + proc * 50; // Slight offset per process
            
            setTimeout(() => {
                if (docIndex < docs.length) {
                    const doc = docs[docIndex];
                    doc.classList.add('mpi-processing', 'processing');
                    
                    setTimeout(() => {
                        doc.classList.remove('mpi-processing', 'processing');
                        doc.classList.add('completed');
                        processedCount++;
                        updateDemoStats('mpi', processedCount, docs.length, Date.now() - demoStartTime);
                        
                        if (processedCount >= docs.length) {
                            updateFinalStats('mpi');
                        }
                    }, processTime);
                }
            }, delay);
        }
    }
}

// Hybrid processing demonstration
function runHybridDemo(docs) {
    const processes = 2;
    const threadsPerProcess = 4;
    const processTime = 200; // ms per document
    let processedCount = 0;
    
    // Distribute documents across processes, then threads within each process
    const docsPerProcess = Math.ceil(docs.length / processes);
    
    for (let proc = 0; proc < processes; proc++) {
        const startIndex = proc * docsPerProcess;
        const endIndex = Math.min(startIndex + docsPerProcess, docs.length);
        const processDocCount = endIndex - startIndex;
        
        // Within each process, use threads
        const batchesInProcess = Math.ceil(processDocCount / threadsPerProcess);
        
        for (let batch = 0; batch < batchesInProcess; batch++) {
            const batchStartIndex = startIndex + (batch * threadsPerProcess);
            const batchEndIndex = Math.min(batchStartIndex + threadsPerProcess, endIndex);
            
            setTimeout(() => {
                // Process all documents in this batch simultaneously (threads)
                for (let i = batchStartIndex; i < batchEndIndex; i++) {
                    const doc = docs[i];
                    doc.classList.add('hybrid-processing', 'processing');
                    
                    setTimeout(() => {
                        doc.classList.remove('hybrid-processing', 'processing');
                        doc.classList.add('completed');
                        processedCount++;
                        updateDemoStats('hybrid', processedCount, docs.length, Date.now() - demoStartTime);
                        
                        if (processedCount >= docs.length) {
                            updateFinalStats('hybrid');
                        }
                    }, processTime);
                }
            }, batch * (processTime + 50) + proc * 100);
        }
    }
}

// Update demo statistics display
function updateDemoStats(version, processed, total, elapsed) {
    const timeElement = document.getElementById('demo-time');
    const docsElement = document.getElementById('demo-docs');
    const unitsElement = document.getElementById('demo-units');
    const efficiencyElement = document.getElementById('demo-efficiency');
    
    if (timeElement) timeElement.textContent = `${elapsed}ms`;
    if (docsElement) docsElement.textContent = processed;
    
    // Set parallel units based on version
    let units = 1;
    switch (version) {
        case 'serial': units = 1; break;
        case 'openmp': units = 4; break;
        case 'mpi': units = 4; break;
        case 'hybrid': units = 8; break;
    }
    
    if (unitsElement) unitsElement.textContent = units;
    
    // Calculate efficiency (simplified)
    let efficiency = 100;
    if (processed > 0 && elapsed > 0) {
        const serialTime = processed * 400; 
        const speedup = serialTime / elapsed;
        efficiency = Math.min((speedup / units) * 100, 100);
        
        // Adjust for realistic overhead
        switch (version) {
            case 'openmp': efficiency *= 0.85; break;
            case 'mpi': efficiency *= 0.75; break;
            case 'hybrid': efficiency *= 0.88; break;
        }
    }
    
    if (efficiencyElement) efficiencyElement.textContent = `${Math.round(efficiency)}%`;
}

function updateFinalStats(version) {
    console.log(`${version} demo completed`);
    // Final stats are already updated in updateDemoStats
}
    
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
    console.log("Resetting demo");
    
    if (demoInterval) {
        clearInterval(demoInterval);
        demoInterval = null;
    }
    
    // Reset all document boxes
    const docs = document.querySelectorAll('.demo-doc, .doc-box');
    docs.forEach(doc => {
        doc.classList.remove('processing', 'completed', 'serial-processing', 'openmp-processing', 'mpi-processing', 'hybrid-processing');
    });
    
    // Reset demo stats
    updateDemoStats('serial', 0, 32, 0);
    demoVersion = '';
}

// Setup interactive controls for demos
function setupInteractiveControls() {
    // Demo control buttons
    const serialBtn = document.querySelector('[data-version="serial"]');
    const openmpBtn = document.querySelector('[data-version="openmp"]');
    const mpiBtn = document.querySelector('[data-version="mpi"]');
    const hybridBtn = document.querySelector('[data-version="hybrid"]');
    const resetBtn = document.getElementById('reset-demo');
    
    if (serialBtn) {
        serialBtn.addEventListener('click', () => startDemo('serial'));
    }
    
    if (openmpBtn) {
        openmpBtn.addEventListener('click', () => startDemo('openmp'));
    }
    
    if (mpiBtn) {
        mpiBtn.addEventListener('click', () => startDemo('mpi'));
    }
    
    if (hybridBtn) {
        hybridBtn.addEventListener('click', () => startDemo('hybrid'));
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetDemo);
    }
}

// Update description
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
    const processingTime = 200;
    
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
window.highlightParallelComponents = function(version) {
    console.log(`Highlighting ${version} parallel components`);
    
    // Remove existing highlights
    document.querySelectorAll('.highlighted').forEach(el => {
        el.classList.remove('highlighted');
    });
    
    // Add highlights based on version
    const versionElements = document.querySelectorAll(`.${version}-color, .${version}-version`);
    versionElements.forEach(el => {
        el.classList.add('highlighted');
        setTimeout(() => {
            el.classList.remove('highlighted');
        }, 3000);
    });
};
