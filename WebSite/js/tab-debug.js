/**
 * Debug utility to check for Bootstrap and tab initialization issues
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log("Starting tab debug checks...");
    
    // Check if Bootstrap is loaded
    if (typeof bootstrap === 'undefined') {
        console.error("Bootstrap is not loaded! Make sure bootstrap.bundle.min.js is included.");
    } else {
        console.log("Bootstrap is loaded. Version available:", bootstrap.Collapse ? "5.x" : "Unknown");
        
        // Check if Tab component exists
        if (typeof bootstrap.Tab === 'undefined') {
            console.error("Bootstrap Tab component is not available!");
        } else {
            console.log("Bootstrap Tab component is available.");
        }
    }
    
    // Check for tab triggers
    const tabTriggers = document.querySelectorAll('[data-bs-toggle="tab"]');
    console.log(`Found ${tabTriggers.length} tab triggers in the document.`);
    
    // Check for tab content panes
    const tabPanes = document.querySelectorAll('.tab-pane');
    console.log(`Found ${tabPanes.length} tab content panes.`);
    
    // Check for active tabs
    const activeTabs = document.querySelectorAll('.tab-pane.active, .tab-pane.show.active');
    console.log(`Found ${activeTabs.length} active tab content panes.`);
    if (activeTabs.length > 0) {
        activeTabs.forEach(tab => {
            console.log(`  Active tab: #${tab.id}`);
        });
    }
    
    // Check for tab container
    const tabContainers = document.querySelectorAll('.tab-content');
    console.log(`Found ${tabContainers.length} tab containers.`);
    
    // List potential issues
    let issues = [];
    
    if (tabTriggers.length === 0) {
        issues.push("No tab triggers found. Check for elements with data-bs-toggle='tab'");
    }
    
    if (tabPanes.length === 0) {
        issues.push("No tab panes found. Check for elements with class 'tab-pane'");
    }
    
    if (activeTabs.length === 0) {
        issues.push("No active tab panes found. At least one tab should have 'active' class");
    }
    
    if (tabTriggers.length > 0 && tabPanes.length > 0) {
        // Check if each tab trigger has a corresponding tab pane
        tabTriggers.forEach(trigger => {
            const target = trigger.getAttribute('href');
            if (!target) {
                issues.push(`Tab trigger is missing href attribute: ${trigger.outerHTML}`);
            } else if (!document.querySelector(target)) {
                issues.push(`Tab pane not found for trigger targeting: ${target}`);
            }
        });
    }
    
    if (issues.length > 0) {
        console.warn("Potential tab issues detected:");
        issues.forEach((issue, index) => {
            console.warn(`  ${index + 1}. ${issue}`);
        });
    } else {
        console.log("No obvious tab issues detected.");
    }
    
    console.log("Tab debug checks complete.");
});
