/**
 * tab-navigator.js
 * Advanced tab navigation functionality for the Search Engine Dashboard
 */

class TabNavigator {
    constructor() {
        this.tabs = [];
        this.currentTab = null;
        this.tabHistory = [];
        this.maxHistory = 20;
        this.listeners = {};
        this.tabStack = [];
    }

    /**
     * Initialize tab navigation
     * @param {Array} tabIds - Array of tab IDs to manage
     * @param {string} defaultTabId - ID of the default tab to show
     */
    init(tabIds, defaultTabId) {
        this.tabs = tabIds;
        
        // Set up tab click handlers
        tabIds.forEach(tabId => {
            const tabElement = document.getElementById(`tab-${tabId}`);
            const contentElement = document.getElementById(`content-${tabId}`);
            
            if (tabElement && contentElement) {
                tabElement.addEventListener('click', () => this.showTab(tabId));
            }
        });

        // Show default tab
        if (defaultTabId && this.tabs.includes(defaultTabId)) {
            this.showTab(defaultTabId);
        } else if (this.tabs.length > 0) {
            this.showTab(this.tabs[0]);
        }

        // Set up keyboard navigation
        document.addEventListener('keydown', (e) => {
            // Ctrl+Tab for next tab
            if (e.ctrlKey && e.key === 'Tab') {
                e.preventDefault();
                this.navigateToNextTab(e.shiftKey ? -1 : 1);
            }
            
            // Alt+number for direct tab access
            if (e.altKey && !isNaN(parseInt(e.key)) && parseInt(e.key) > 0) {
                const index = parseInt(e.key) - 1;
                if (index < this.tabs.length) {
                    e.preventDefault();
                    this.showTab(this.tabs[index]);
                }
            }
        });
    }

    /**
     * Show a specific tab
     * @param {string} tabId - ID of tab to show
     */
    showTab(tabId) {
        if (!this.tabs.includes(tabId)) {
            console.error(`Tab '${tabId}' not found in registered tabs`);
            return false;
        }

        // Hide all tabs
        this.tabs.forEach(id => {
            const tabElement = document.getElementById(`tab-${id}`);
            const contentElement = document.getElementById(`content-${id}`);
            
            if (tabElement) tabElement.classList.remove('active');
            if (contentElement) contentElement.style.display = 'none';
        });

        // Show selected tab
        const tabElement = document.getElementById(`tab-${tabId}`);
        const contentElement = document.getElementById(`content-${tabId}`);
        
        if (tabElement) tabElement.classList.add('active');
        if (contentElement) contentElement.style.display = 'block';

        // Update history
        if (this.currentTab !== null && this.currentTab !== tabId) {
            this.tabHistory.unshift(this.currentTab);
            if (this.tabHistory.length > this.maxHistory) {
                this.tabHistory.pop();
            }
        }
        
        const previousTab = this.currentTab;
        this.currentTab = tabId;
        
        // Trigger events
        this.triggerEvent('tabChanged', { 
            currentTab: tabId, 
            previousTab: previousTab 
        });

        return true;
    }

    /**
     * Get the ID of the currently visible tab
     * @returns {string} Current tab ID
     */
    getCurrentTab() {
        return this.currentTab;
    }

    /**
     * Navigate to the previous tab in history
     * @returns {boolean} True if navigation successful
     */
    navigateBack() {
        if (this.tabHistory.length === 0) {
            return false;
        }
        
        const prevTab = this.tabHistory.shift();
        if (prevTab) {
            // Don't record this navigation in history
            const currentTab = this.currentTab;
            this.showTab(prevTab);
            // Remove the automatic history entry and restore proper order
            this.tabHistory.shift();
            this.tabHistory.unshift(currentTab);
            return true;
        }
        
        return false;
    }

    /**
     * Navigate to the next or previous tab in the tab list
     * @param {number} direction - Direction to navigate (1 for next, -1 for previous)
     */
    navigateToNextTab(direction = 1) {
        if (!this.currentTab || this.tabs.length <= 1) return;
        
        const currentIndex = this.tabs.indexOf(this.currentTab);
        if (currentIndex === -1) return;
        
        let nextIndex = currentIndex + direction;
        
        // Loop around if we go past the ends
        if (nextIndex >= this.tabs.length) {
            nextIndex = 0;
        } else if (nextIndex < 0) {
            nextIndex = this.tabs.length - 1;
        }
        
        this.showTab(this.tabs[nextIndex]);
    }

    /**
     * Push a tab onto the navigation stack
     * @param {string} tabId - ID of tab to push
     */
    pushTab(tabId) {
        if (this.currentTab) {
            this.tabStack.push(this.currentTab);
        }
        this.showTab(tabId);
    }

    /**
     * Pop a tab from the navigation stack
     */
    popTab() {
        if (this.tabStack.length > 0) {
            const previousTab = this.tabStack.pop();
            this.showTab(previousTab);
        }
    }

    /**
     * Add event listener
     * @param {string} event - Event name ('tabChanged')
     * @param {Function} callback - Callback function
     */
    addEventListener(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function to remove
     */
    removeEventListener(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
    }

    /**
     * Trigger an event
     * @param {string} event - Event name
     * @param {Object} data - Event data
     */
    triggerEvent(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in tab navigator event listener (${event}):`, error);
                }
            });
        }
    }
}

// Create singleton instance
const tabNavigator = new TabNavigator();

export default tabNavigator;
