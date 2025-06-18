/**
 * Tab navigation utility to safely navigate between tabs
 * This provides a safer alternative to directly manipulating the tab DOM
 */

// Safe tab navigation function that can be called from anywhere in the code
function navigateToTab(tabId) {
    if (!tabId) return false;
    
    // Ensure tabId starts with #
    if (!tabId.startsWith('#')) {
        tabId = '#' + tabId;
    }

    try {
        // Get the tab trigger that targets this tab
        const tabTrigger = document.querySelector(`[data-bs-toggle="tab"][href="${tabId}"]`);
        if (!tabTrigger) {
            console.warn(`No tab trigger found for tab ID: ${tabId}`);
            return false;
        }
        
        // Try to use Bootstrap's Tab API first if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tab) {
            const bsTab = new bootstrap.Tab(tabTrigger);
            bsTab.show();
            return true;
        }
        
        // Fallback: trigger a click event on the tab
        tabTrigger.click();
        return true;
    }
    catch (err) {
        console.error(`Error navigating to tab ${tabId}:`, err);
        return false;
    }
}

// Function to check if all tabs are working correctly
function verifyTabSystem() {
    const tabPanes = document.querySelectorAll('.tab-pane');
    const tabTriggers = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    console.log(`Tab system verification: Found ${tabPanes.length} tab panes and ${tabTriggers.length} tab triggers`);
    
    // Check if all tab triggers have corresponding panes
    let allValid = true;
    tabTriggers.forEach(trigger => {
        const targetId = trigger.getAttribute('href');
        if (!targetId) {
            console.error(`Tab trigger has no target: ${trigger.outerHTML}`);
            allValid = false;
            return;
        }
        
        const targetPane = document.querySelector(targetId);
        if (!targetPane) {
            console.error(`Tab pane not found for ${targetId}`);
            allValid = false;
        }
    });
    
    return allValid;
}

// Auto-run verification when included in a page
document.addEventListener('DOMContentLoaded', function() {
    // Wait for Bootstrap to be fully loaded
    setTimeout(() => {
        verifyTabSystem();
    }, 500);
});
