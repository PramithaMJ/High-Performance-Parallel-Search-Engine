/**
 * error-monitor.js
 * Error monitoring and handling for the Search Engine Dashboard
 */

class ErrorMonitor {
    constructor() {
        this.errors = [];
        this.maxErrors = 50;  // Maximum number of errors to keep in history
        this.listeners = [];
        this.setupGlobalErrorHandler();
    }

    /**
     * Initialize error monitoring
     */
    init() {
        console.log("Error monitor initialized");
        this.loadErrorsFromStorage();
    }

    /**
     * Set up global error event listeners
     */
    setupGlobalErrorHandler() {
        window.addEventListener('error', (event) => {
            this.logError('Uncaught Error', event.error.message, event.error.stack);
            return false;
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.logError('Unhandled Promise Rejection', event.reason.message || String(event.reason), 
                          event.reason.stack || 'No stack trace');
            return false;
        });
    }

    /**
     * Log an error to the error monitoring system
     * @param {string} type - The type of error
     * @param {string} message - Error message
     * @param {string} stack - Stack trace (optional)
     * @param {Object} details - Additional error details (optional)
     */
    logError(type, message, stack = '', details = {}) {
        const error = {
            id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
            timestamp: new Date().toISOString(),
            type: type,
            message: message,
            stack: stack,
            details: details
        };

        this.errors.unshift(error);
        if (this.errors.length > this.maxErrors) {
            this.errors.pop();
        }

        this.saveErrorsToStorage();
        this.notifyListeners(error);
        
        console.error(`[${type}] ${message}`);
        return error.id;
    }

    /**
     * Log an API error
     * @param {string} endpoint - The API endpoint that failed
     * @param {Object} response - The error response
     */
    logApiError(endpoint, response) {
        let message = 'API Error';
        let details = { endpoint };
        
        if (response) {
            if (response.status) details.status = response.status;
            
            try {
                if (response.json) {
                    response.json().then(data => {
                        details.response = data;
                        this.logError('API Error', 
                            `${endpoint} failed with status ${response.status}`, 
                            null, details);
                    }).catch(e => {
                        details.responseText = response.statusText;
                        this.logError('API Error', 
                            `${endpoint} failed with status ${response.status}`, 
                            null, details);
                    });
                    return;
                }
            } catch (e) {
                details.parseError = e.message;
            }
        }
        
        this.logError('API Error', `${endpoint} request failed`, null, details);
    }

    /**
     * Get all recorded errors
     */
    getErrors() {
        return [...this.errors];
    }

    /**
     * Clear all recorded errors
     */
    clearErrors() {
        this.errors = [];
        this.saveErrorsToStorage();
        this.notifyListeners(null);
    }

    /**
     * Save errors to local storage
     */
    saveErrorsToStorage() {
        try {
            localStorage.setItem('searchEngineErrors', JSON.stringify(this.errors));
        } catch (e) {
            console.warn('Failed to save errors to local storage:', e);
        }
    }

    /**
     * Load errors from local storage
     */
    loadErrorsFromStorage() {
        try {
            const savedErrors = localStorage.getItem('searchEngineErrors');
            if (savedErrors) {
                this.errors = JSON.parse(savedErrors);
                if (!Array.isArray(this.errors)) {
                    this.errors = [];
                }
            }
        } catch (e) {
            console.warn('Failed to load errors from local storage:', e);
            this.errors = [];
        }
    }

    /**
     * Add an error listener
     * @param {Function} callback - Function to call when an error occurs
     */
    addListener(callback) {
        if (typeof callback === 'function' && !this.listeners.includes(callback)) {
            this.listeners.push(callback);
        }
        return this;
    }

    /**
     * Remove an error listener
     * @param {Function} callback - Function to remove from listeners
     */
    removeListener(callback) {
        this.listeners = this.listeners.filter(listener => listener !== callback);
        return this;
    }

    /**
     * Notify all listeners of a new error
     * @param {Object} error - The error object
     */
    notifyListeners(error) {
        this.listeners.forEach(listener => {
            try {
                listener(error);
            } catch (e) {
                console.error('Error in error listener:', e);
            }
        });
    }
}

// Create a singleton instance
const errorMonitor = new ErrorMonitor();
errorMonitor.init();

export default errorMonitor;
