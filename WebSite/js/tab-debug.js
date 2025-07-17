/**
 * tab-debug.js
 * Debugging utilities for tab navigation and tab content in the Search Engine Dashboard
 */

import errorMonitor from './error-monitor.js';

class TabDebugger {
    constructor() {
        this.isEnabled = false;
        this.logHistory = [];
        this.maxLogEntries = 100;
        this.debugOverlay = null;
        this.activeTimers = new Map();
    }

    /**
     * Initialize the tab debugger
     * @param {boolean} enabled - Whether debugging is initially enabled
     */
    init(enabled = false) {
        this.isEnabled = enabled;
        
        // Create debug overlay if enabled
        if (enabled) {
            this.createDebugOverlay();
        }
        
        // Set up keyboard shortcut (Ctrl+Shift+D) to toggle debug mode
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.toggleDebugMode();
            }
        });

        console.log("Tab debugger initialized, press Ctrl+Shift+D to toggle");
    }

    /**
     * Toggle debug mode on/off
     */
    toggleDebugMode() {
        this.isEnabled = !this.isEnabled;
        
        if (this.isEnabled) {
            this.createDebugOverlay();
            this.log('Debug mode enabled');
        } else {
            this.removeDebugOverlay();
            this.log('Debug mode disabled');
        }
        
        // Save debug mode preference
        try {
            localStorage.setItem('tabDebugEnabled', this.isEnabled);
        } catch (e) {
            console.warn('Failed to save debug mode preference:', e);
        }
        
        return this.isEnabled;
    }

    /**
     * Create debug overlay UI
     */
    createDebugOverlay() {
        if (this.debugOverlay) {
            return;
        }
        
        // Create overlay container
        this.debugOverlay = document.createElement('div');
        this.debugOverlay.id = 'tab-debug-overlay';
        this.debugOverlay.className = 'debug-overlay';
        
        // Style the overlay
        Object.assign(this.debugOverlay.style, {
            position: 'fixed',
            bottom: '10px',
            right: '10px',
            width: '300px',
            maxHeight: '400px',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: '#00ff00',
            fontFamily: 'monospace',
            fontSize: '12px',
            padding: '10px',
            borderRadius: '5px',
            zIndex: '9999',
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column'
        });
        
        // Create header
        const header = document.createElement('div');
        header.textContent = 'Tab Debugger';
        header.style.marginBottom = '5px';
        header.style.fontWeight = 'bold';
        header.style.borderBottom = '1px solid #00ff00';
        header.style.paddingBottom = '5px';
        
        // Create log container
        const logContainer = document.createElement('div');
        logContainer.id = 'tab-debug-log';
        logContainer.style.overflowY = 'auto';
        logContainer.style.flex = '1';
        
        // Create controls
        const controls = document.createElement('div');
        controls.style.marginTop = '5px';
        controls.style.borderTop = '1px solid #00ff00';
        controls.style.paddingTop = '5px';
        
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.onclick = () => this.clearLog();
        
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.onclick = () => this.toggleDebugMode();
        
        controls.appendChild(clearBtn);
        controls.appendChild(closeBtn);
        
        // Assemble overlay
        this.debugOverlay.appendChild(header);
        this.debugOverlay.appendChild(logContainer);
        this.debugOverlay.appendChild(controls);
        
        // Add to document
        document.body.appendChild(this.debugOverlay);
        
        // Make draggable
        this.makeElementDraggable(this.debugOverlay);
        
        // Display existing logs
        this.refreshLogDisplay();
    }

    /**
     * Make an element draggable
     * @param {HTMLElement} element - Element to make draggable
     */
    makeElementDraggable(element) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        
        const header = element.firstChild;
        header.style.cursor = 'move';
        header.onmousedown = dragMouseDown;
        
        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            // Get mouse position at startup
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            // Call function when mouse moves
            document.onmousemove = elementDrag;
        }
        
        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            // Calculate new cursor position
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            // Set new position
            element.style.top = (element.offsetTop - pos2) + "px";
            element.style.left = (element.offsetLeft - pos1) + "px";
            // Make sure it stays on screen
            if (parseInt(element.style.top) < 0) element.style.top = "0px";
            if (parseInt(element.style.left) < 0) element.style.left = "0px";
        }
        
        function closeDragElement() {
            // Stop moving when mouse button released
            document.onmouseup = null;
            document.onmousemove = null;
        }
    }

    /**
     * Remove the debug overlay
     */
    removeDebugOverlay() {
        if (this.debugOverlay && this.debugOverlay.parentNode) {
            this.debugOverlay.parentNode.removeChild(this.debugOverlay);
            this.debugOverlay = null;
        }
    }

    /**
     * Log a message to the debug console
     * @param {string} message - Message to log
     * @param {string} type - Log type (info, warn, error)
     */
    log(message, type = 'info') {
        if (!message) return;
        
        const entry = {
            timestamp: new Date(),
            message: message.toString(),
            type: type
        };
        
        this.logHistory.unshift(entry);
        
        // Limit log history size
        if (this.logHistory.length > this.maxLogEntries) {
            this.logHistory.pop();
        }
        
        // Update display if visible
        if (this.isEnabled && this.debugOverlay) {
            this.refreshLogDisplay();
        }
        
        // Always log to console
        switch (type) {
            case 'error':
                console.error('[TabDebug]', message);
                break;
            case 'warn':
                console.warn('[TabDebug]', message);
                break;
            default:
                console.log('[TabDebug]', message);
        }
    }

    /**
     * Clear the log history
     */
    clearLog() {
        this.logHistory = [];
        this.refreshLogDisplay();
    }

    /**
     * Refresh the log display in the UI
     */
    refreshLogDisplay() {
        if (!this.debugOverlay) return;
        
        const logContainer = document.getElementById('tab-debug-log');
        if (!logContainer) return;
        
        // Clear existing content
        logContainer.innerHTML = '';
        
        if (this.logHistory.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.textContent = 'No log entries';
            emptyMessage.style.fontStyle = 'italic';
            emptyMessage.style.color = '#888';
            logContainer.appendChild(emptyMessage);
            return;
        }
        
        // Add log entries
        this.logHistory.forEach(entry => {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${entry.type}`;
            logEntry.style.marginBottom = '3px';
            logEntry.style.wordBreak = 'break-word';
            
            // Color based on type
            switch (entry.type) {
                case 'error':
                    logEntry.style.color = '#ff5555';
                    break;
                case 'warn':
                    logEntry.style.color = '#ffff55';
                    break;
                default:
                    logEntry.style.color = '#00ff00';
            }
            
            // Format timestamp
            const time = entry.timestamp.toLocaleTimeString('en-US', { 
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            
            logEntry.textContent = `[${time}] ${entry.message}`;
            logContainer.appendChild(logEntry);
        });
        
        // Scroll to top
        logContainer.scrollTop = 0;
    }

    /**
     * Start a performance timer
     * @param {string} label - Timer label
     */
    startTimer(label) {
        if (!label) return;
        
        this.activeTimers.set(label, performance.now());
        this.log(`Timer started: ${label}`);
    }

    /**
     * End a performance timer and return elapsed time
     * @param {string} label - Timer label
     * @param {boolean} logResult - Whether to log the result
     * @returns {number} Elapsed milliseconds
     */
    endTimer(label, logResult = true) {
        if (!label || !this.activeTimers.has(label)) {
            this.log(`Timer '${label}' not found`, 'warn');
            return 0;
        }
        
        const startTime = this.activeTimers.get(label);
        const elapsed = performance.now() - startTime;
        
        this.activeTimers.delete(label);
        
        if (logResult) {
            this.log(`Timer '${label}': ${elapsed.toFixed(2)}ms`);
        }
        
        return elapsed;
    }

    /**
     * Log information about a tab change event
     * @param {string} fromTab - Previous tab ID
     * @param {string} toTab - New tab ID
     */
    logTabChange(fromTab, toTab) {
        this.log(`Tab change: ${fromTab || 'null'} → ${toTab}`);
    }

    /**
     * Log an error that occurred in a specific tab
     * @param {string} tabId - ID of the tab where the error occurred
     * @param {string} message - Error message
     * @param {Object} details - Error details
     */
    logTabError(tabId, message, details = {}) {
        this.log(`Error in tab ${tabId}: ${message}`, 'error');
        errorMonitor.logError(`Tab ${tabId} Error`, message, null, details);
    }
}

// Create singleton instance
const tabDebugger = new TabDebugger();

export default tabDebugger;
