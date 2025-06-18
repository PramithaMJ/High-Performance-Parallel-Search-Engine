/**
 * Bootstrap tab error fix utility 
 * This script addresses the "Uncaught TypeError: Illegal invocation" error in selector-engine.js
 * and ensures tabs are working correctly
 */

(function() {
    // Run this after the page is fully loaded and Bootstrap should be available
    window.addEventListener('load', function() {
        console.log("Bootstrap tab fix utility loaded");

        // Step 1: Ensure Bootstrap is properly loaded
        if (typeof bootstrap === 'undefined') {
            console.error("Bootstrap is not defined. Adding it again.");
            loadBootstrap();
            return; // The loadBootstrap function will call fixTabs after loading
        } else {
            console.log("Bootstrap found:", bootstrap);
            // Give a short delay to ensure all components are registered
            setTimeout(fixTabs, 100);
        }
    });

    // Load Bootstrap if it's missing
    function loadBootstrap() {
        const script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js";
        script.onload = function() {
            console.log("Bootstrap loaded dynamically");
            setTimeout(fixTabs, 100);
        };
        script.onerror = function() {
            console.error("Failed to load Bootstrap dynamically");
        };
        document.body.appendChild(script);
    }

    // Fix tab navigation issues
    function fixTabs() {
        // Step 2: Manually initialize tabs to avoid the selector-engine.js error
        const tabElements = document.querySelectorAll('[data-bs-toggle="tab"]');
        console.log(`Found ${tabElements.length} tab elements`);
        
        if (tabElements.length === 0) {
            console.warn("No tab elements found");
            return;
        }
        
        // Remove any existing click handlers that might cause issues
        tabElements.forEach(tab => {
            // Clone the element to remove all event listeners
            const newTab = tab.cloneNode(true);
            tab.parentNode.replaceChild(newTab, tab);
        });
            
        // Get the refreshed tab elements after cloning
        const refreshedTabs = document.querySelectorAll('[data-bs-toggle="tab"]');
        
        // Re-initialize all tabs with safer code
        refreshedTabs.forEach(tab => {
            tab.addEventListener('click', function(event) {
                event.preventDefault();
                
                try {
                    // Get the target tab pane ID from the href attribute
                    const targetId = this.getAttribute('href');
                    if (!targetId) {
                        console.warn("Tab has no href attribute");
                        return;
                    }
                    
                    // Get the target tab pane
                    const targetPane = document.querySelector(targetId);
                    if (!targetPane) {
                        console.warn(`Target tab pane not found: ${targetId}`);
                        return;
                    }
                    
                    // Find all tab panes in the same container
                    const tabContainer = targetPane.parentNode;
                    if (!tabContainer) {
                        console.warn("Tab container not found");
                        return;
                    }
                    
                    // Manually handle the tab switching
                    const allPanes = tabContainer.querySelectorAll('.tab-pane');
                    allPanes.forEach(pane => {
                        pane.classList.remove('show', 'active');
                    });
                    
                    // Activate the target pane
                    targetPane.classList.add('show', 'active');
                    
                    // Update the active state on the tabs
                    const tabContainer2 = this.closest('.nav');
                    if (tabContainer2) {
                        const allTabs = tabContainer2.querySelectorAll('[data-bs-toggle="tab"]');
                        allTabs.forEach(t => {
                            t.classList.remove('active');
                            const parent = t.parentNode;
                            if (parent && parent.tagName === 'LI') {
                                parent.classList.remove('active');
                            }
                        });
                        
                        this.classList.add('active');
                        const parent = this.parentNode;
                        if (parent && parent.tagName === 'LI') {
                            parent.classList.add('active');
                        }
                    }
                    
                    // Log success message
                    console.log(`Successfully switched to tab: ${targetId}`);
                } catch (error) {
                    console.error("Error handling tab click:", error);
                }
            });
        });
        
        // Ensure the active tab is properly shown
        const activeTabLink = document.querySelector('[data-bs-toggle="tab"].active');
        if (activeTabLink) {
            console.log("Found active tab link:", activeTabLink);
            // The tab is already marked as active in HTML, just make sure the pane is visible
            const targetId = activeTabLink.getAttribute('href');
            if (targetId) {
                const targetPane = document.querySelector(targetId);
                if (targetPane) {
                    // Ensure all other panes in the same container are hidden
                    const tabContainer = targetPane.parentNode;
                    if (tabContainer) {
                        const allPanes = tabContainer.querySelectorAll('.tab-pane');
                        allPanes.forEach(pane => {
                            if (pane !== targetPane) {
                                pane.classList.remove('show', 'active');
                            }
                        });
                    }
                    // Make sure this pane is visible
                    targetPane.classList.add('show', 'active');
                    console.log(`Activated tab pane: ${targetId}`);
                }
            }
        } else {
            // Find the first tab in each tab container and activate it
            const tabContainers = document.querySelectorAll('.nav-tabs, .nav-pills');
            tabContainers.forEach(container => {
                const firstTab = container.querySelector('[data-bs-toggle="tab"]');
                if (firstTab) {
                    console.log("Activating first tab in container:", container);
                    firstTab.click();
                }
            });
        }
    }
})();
