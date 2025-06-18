/**
 * This utility handles proper tab activation and navigation in the dashboard.
 * It fixes the "Uncaught TypeError: Illegal invocation" error in the selector-engine.js
 * by properly initializing Bootstrap tabs.
 * 
 * IMPORTANT: This version uses a more defensive programming approach to avoid the
 * "Illegal invocation" error and ensures proper tab functionality even if Bootstrap
 * is not fully loaded.
 */

// Create a tab initialization function
function initializeTabSystem() {
    console.log("Tab system initialization started...");
    
    // Check if jQuery is available first (historically used by Bootstrap)
    const jQueryAvailable = typeof $ !== 'undefined' && typeof $.fn !== 'undefined';
    if (jQueryAvailable) {
        console.log("jQuery detected - version:", $.fn.jquery);
    } else {
        console.log("jQuery not detected, using vanilla JavaScript");
    }
    
    // Initialize all Bootstrap tabs using the Bootstrap Tab API
    // First check if Bootstrap is available
    if (typeof bootstrap === 'undefined' || !bootstrap.Tab) {
        console.warn('Bootstrap Tab API not found. Using manual tab activation as fallback.');
    } else {
        console.log("Bootstrap Tab API found - attempting to use it safely");
    }

    // Safely get the tab trigger elements
    let tabTriggerList = [];
    try {
        // Use safe method to get tab triggers
        const tabElements = document.querySelectorAll('[data-bs-toggle="tab"]');
        tabTriggerList = Array.prototype.slice.call(tabElements);
        console.log(`Found ${tabTriggerList.length} tab trigger elements`);
    } catch (err) {
        console.warn(`Error selecting tab triggers: ${err.message}`);
        return; // Exit early if we can't even select tabs
    }

    // Step 1: First try to initialize with Bootstrap if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tab) {
        try {
            console.log("Attempting to initialize tabs with Bootstrap Tab API");
            // Use a safe approach to create Tab instances
            tabTriggerList.forEach(function(tabTriggerEl) {
                try {
                    // Store the tab instance on the element itself
                    tabTriggerEl._bsTab = new bootstrap.Tab(tabTriggerEl);
                } catch (err) {
                    console.warn(`Error initializing Bootstrap tab for ${tabTriggerEl.getAttribute('href')}: ${err.message}`);
                }
            });
        } catch (err) {
            console.warn(`Error during Bootstrap tab initialization: ${err.message}`);
        }
    }
    
    // Step 2: Add safe click handlers to all tab triggers regardless of Bootstrap availability
    console.log("Setting up safe click handlers for tab navigation");
    tabTriggerList.forEach(function(tabTrigger) {
        // Remove existing listeners by cloning the element
        const newTabTrigger = tabTrigger.cloneNode(true);
        if (tabTrigger.parentNode) {
            tabTrigger.parentNode.replaceChild(newTabTrigger, tabTrigger);
            // Add our safe handler to the new element
            newTabTrigger.addEventListener('click', safeHandleTabClick);
        }
    });
    
    // Step 3: Make sure at least one tab is active
    setTimeout(function() {
        const activeTabPane = document.querySelector('.tab-pane.active');
        if (!activeTabPane) {
            console.log("No active tab found. Activating the first tab...");
            const firstTab = document.querySelector('[data-bs-toggle="tab"]');
            if (firstTab) {
                firstTab.click();
            }
        } else {
            console.log("Active tab found:", activeTabPane.id);
        }
    }, 100);
    
    // Safe click handler function that prevents the "Illegal invocation" error
    function safeHandleTabClick(event) {
        event.preventDefault();
        
        try {
            // Get essential information first
            const tabTrigger = this;
            const href = tabTrigger.getAttribute('href');
            
            console.log(`Tab clicked: ${href}`);
            
            // Method 1: Try using the Bootstrap Tab API if available
            if (typeof bootstrap !== 'undefined' && bootstrap.Tab) {
                try {
                    // Use the stored instance if available, or create a new one
                    const tabInstance = tabTrigger._bsTab || new bootstrap.Tab(tabTrigger);
                    tabInstance.show();
                    console.log(`Tab activated via Bootstrap API: ${href}`);
                    return; // Success! No need to try other methods
                } catch (bsError) {
                    console.warn(`Bootstrap tab activation failed: ${bsError.message}. Falling back to manual activation.`);
                    // Continue to fallback method
                }
            }
            
            // Method 2: Manual activation
            // Get tab target from href
            if (!href) {
                console.warn('Tab trigger has no href attribute');
                return;
            }
            
            const targetTab = document.querySelector(href);
            if (!targetTab) {
                console.warn(`Target tab not found: ${href}`);
                return;
            }
            
            // Find tab container
            const tabContent = targetTab.closest('.tab-content');
            if (!tabContent) {
                console.warn(`Tab content container not found for: ${href}`);
                return;
            }
            
            // Deactivate all tabs in this container
            const allTabPanes = tabContent.querySelectorAll('.tab-pane');
            allTabPanes.forEach(pane => {
                pane.classList.remove('show', 'active');
            });
            
            // Activate target tab
            targetTab.classList.add('show', 'active');
            
            // Update nav tabs active state
            const navTab = tabTrigger.closest('.nav-tabs, .nav-pills, .nav');
            if (navTab) {
                // Remove active class from all tabs
                const allNavItems = navTab.querySelectorAll('[data-bs-toggle="tab"]');
                allNavItems.forEach(navItem => {
                    navItem.classList.remove('active');
                    const parentLi = navItem.closest('li');
                    if (parentLi) {
                        parentLi.classList.remove('active');
                    }
                });
                
                // Set active class on clicked tab
                tabTrigger.classList.add('active');
                const parentLi = tabTrigger.closest('li');
                if (parentLi) {
                    parentLi.classList.add('active');
                }
            }
            
            console.log(`Tab manually activated: ${href}`);
            
            // Dispatch custom event that other code might be listening for
            const tabShownEvent = new CustomEvent('tab.shown', { 
                bubbles: true, 
                detail: { 
                    target: targetTab, 
                    relatedTarget: null 
                } 
            });
            tabTrigger.dispatchEvent(tabShownEvent);
            
        } catch (err) {
            console.error('Error in tab click handler:', err);
        }
    }
    
    // Ensure the default tab is shown
    document.querySelectorAll('.tab-content').forEach(function(tabContent) {
        // Get active tab or first tab as default
        const activeTab = tabContent.querySelector('.tab-pane.active') || 
                         tabContent.querySelector('.tab-pane.show.active') || 
                         tabContent.querySelector('.tab-pane:first-child');
                         
        if (activeTab) {
            // Ensure it has the right classes
            activeTab.classList.add('show', 'active');
            
            // Also activate the corresponding nav item
            const tabId = activeTab.id;
            const correspondingNavItem = document.querySelector(`[href="#${tabId}"]`);
            if (correspondingNavItem) {
                correspondingNavItem.classList.add('active');
                const parentListItem = correspondingNavItem.closest('li');
                if (parentListItem) {
                    parentListItem.classList.add('active');
                }
            }
        }
    });
    
    // Special handling for the search tab to ensure search functionality works properly
    const searchTab = document.getElementById('search-engine');
    if (searchTab) {
        searchTab.addEventListener('tab.shown', function() {
            console.log('Search tab shown - ensuring search functionality is ready');
            // Make sure the search input is focused when this tab is shown
            const searchInput = document.querySelector('.search-input');
            if (searchInput) {
                setTimeout(() => {
                    searchInput.focus();
                }, 200);
            }
        });
    }
}
