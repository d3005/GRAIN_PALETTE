// GrainPalette JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeFileUpload();
    initializeFormSubmission();
    
    // Initialize tooltips if Bootstrap tooltips are available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Enhanced file upload with drag and drop functionality
function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const uploadZone = document.getElementById('uploadZone');
    const imageSize = document.getElementById('imageSize');
    
    if (!fileInput || !imagePreview || !previewImg || !uploadZone) return;
    
    // Drag and drop functionality
    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            handleFileSelection(file);
        }
    });
    
    // Click to upload
    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });
    
    function handleFileSelection(file) {
        // Validate file type
        if (!isValidImageFile(file)) {
            showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WEBP)', 'error');
            resetFileInput();
            return;
        }
        
        // Validate file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showAlert('File size must be less than 16MB', 'error');
            resetFileInput();
            return;
        }
        
        // Update file input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            
            // Update image size info
            if (imageSize) {
                const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
                imageSize.textContent = `${file.name} (${sizeInMB} MB)`;
            }
            
            // Show preview with animation
            imagePreview.style.display = 'block';
            imagePreview.style.opacity = '0';
            imagePreview.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                imagePreview.style.transition = 'all 0.4s ease';
                imagePreview.style.opacity = '1';
                imagePreview.style.transform = 'translateY(0)';
                
                // Smooth scroll to preview
                setTimeout(() => {
                    imagePreview.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'nearest' 
                    });
                }, 200);
            }, 50);
        };
        reader.readAsDataURL(file);
    }
    
    function resetFileInput() {
        fileInput.value = '';
        imagePreview.style.display = 'none';
    }
}

// Remove image function
function removeImage() {
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('imagePreview');
    
    if (fileInput) fileInput.value = '';
    if (imagePreview) {
        imagePreview.style.transition = 'all 0.3s ease';
        imagePreview.style.opacity = '0';
        imagePreview.style.transform = 'translateY(-20px)';
        
        setTimeout(() => {
            imagePreview.style.display = 'none';
        }, 300);
    }
}

// Form submission with loading state
function initializeFormSubmission() {
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingModal = document.getElementById('loadingModal');
    
    if (!uploadForm || !submitBtn) return;
    
    uploadForm.addEventListener('submit', function(event) {
        const fileInput = document.getElementById('file');
        
        // Validate file selection
        if (!fileInput || !fileInput.files.length) {
            event.preventDefault();
            showAlert('Please select an image file to analyze', 'error');
            return;
        }
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Analyzing...
        `;
        
        // Show loading modal if available
        if (loadingModal && typeof bootstrap !== 'undefined') {
            const modal = new bootstrap.Modal(loadingModal);
            modal.show();
        }
    });
}

// Validate image file type
function isValidImageFile(file) {
    const validTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/gif', 'image/bmp', 'image/webp'];
    return validTypes.includes(file.type.toLowerCase());
}

// Show alert message
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'info'} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="feather-${type === 'error' ? 'alert-circle' : 'info'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Scroll to alert
        alertDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    // Re-initialize feather icons for the new alert
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

// Utility functions for results page
function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach((bar, index) => {
        setTimeout(() => {
            bar.style.width = bar.getAttribute('aria-valuenow') + '%';
        }, index * 200);
    });
}

// Initialize progress bar animations on results page
if (document.querySelector('.progress-bar')) {
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(animateProgressBars, 500);
    });
}

// Print functionality
function printResults() {
    window.print();
}

// Copy results to clipboard
async function copyResults() {
    const prediction = document.querySelector('.display-4.text-success');
    const confidence = document.querySelector('.lead strong.text-info');
    
    if (prediction && confidence) {
        const text = `Rice Type: ${prediction.textContent}\nConfidence: ${confidence.textContent}`;
        
        try {
            await navigator.clipboard.writeText(text);
            showAlert('Results copied to clipboard!', 'success');
        } catch (err) {
            showAlert('Failed to copy results', 'error');
        }
    }
}

// Smooth scrolling for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Handle window resize for responsive design
window.addEventListener('resize', function() {
    // Adjust layout if needed for mobile devices
    if (window.innerWidth < 768) {
        const cards = document.querySelectorAll('.card');
        cards.forEach(card => {
            card.classList.add('mb-3');
            card.classList.remove('mb-4', 'mb-5');
        });
    }
});

// Error handling for image loading
document.addEventListener('error', function(e) {
    if (e.target.tagName === 'IMG') {
        console.error('Image failed to load:', e.target.src);
        // Could add fallback image or error message here
    }
}, true);

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(function() {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
        }, 0);
    });
}
