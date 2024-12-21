// Select relevant DOM elements
const sidebar = document.querySelector(".sidebar");
const sidebarToggler = document.querySelector(".sidebar-toggler");
const menuToggler = document.querySelector(".menu-toggler");
const navLinks = document.querySelectorAll(".nav-link");
const collapsedSidebarHeight = "56px";
const maincontainer = document.getElementsByClassName(".main-container");

// Sidebar toggling
sidebarToggler.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    updateTooltips();
});

// Toggle menu height
const toggleMenu = (isMenuActive) => {
    sidebar.style.height = isMenuActive 
        ? `${sidebar.scrollHeight}px` 
        : collapsedSidebarHeight;


    maincontainer.style.setProperty('width', isMenuActive ? 'calc(100% - 316px)' : 'calc(100% - 200px)', 'important');


    menuToggler.querySelector("span").innerText = isMenuActive ? "close" :
     "menu"   
};

menuToggler.addEventListener("click", () => {
    toggleMenu(sidebar.classList.toggle("menu-active"));
});

// Update tooltips based on sidebar state
function updateTooltips() {
    const isCollapsed = sidebar.classList.contains("collapsed");

    // Update tooltip visibility when sidebar is collapsed
    navLinks.forEach((link) => {
        const tooltip = link.getAttribute("data-tooltip");
        if (isCollapsed && tooltip) {
            link.setAttribute("title", tooltip);
        } else {
            link.removeAttribute("title");
        }
    });
}

// Initialize tooltips on page load
updateTooltips();
