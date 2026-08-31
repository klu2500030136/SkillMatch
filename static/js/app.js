document.addEventListener('DOMContentLoaded', () => {
    // Sidebar Toggle for Mobile
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const sidebar = document.getElementById('sidebar');

    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar && sidebar.classList.contains('open')) {
            if (!sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        }
    });

    // File Upload UI Behavior
    const fileInput = document.getElementById('dataset-upload');
    const dropArea = document.getElementById('upload-area');
    const selectedState = document.getElementById('file-selected-state');
    const fileNameDisplay = document.getElementById('file-name');
    const fileSizeDisplay = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const datasetStatus = document.getElementById('dataset-status');
    const datasetStatusText = document.getElementById('dataset-status-text');

    if (fileInput && dropArea) {
        // Handle click on drop area
        dropArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Handle file selection via input
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.remove('dragover');
            }, false);
        });

        dropArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }, false);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            
            // Validate extension
            const validExtensions = ['.csv', '.xlsx'];
            const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
            
            if (!validExtensions.includes(fileExtension)) {
                alert('Please upload a CSV or XLSX file.');
                return;
            }

            // Validate size (10MB max)
            const maxSize = 10 * 1024 * 1024; // 10MB
            if (file.size > maxSize) {
                alert('File size exceeds the 10MB limit.');
                return;
            }

            // Update UI
            showSelectedFile(file.name, file.size);
        }
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function showSelectedFile(name, size) {
        dropArea.style.display = 'none';
        selectedState.style.display = 'flex';
        
        fileNameDisplay.textContent = name;
        fileSizeDisplay.textContent = formatFileSize(size);

        if (datasetStatus) datasetStatus.classList.add('ready');
        if (datasetStatusText) datasetStatusText.textContent = 'Dataset ready';
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', () => {
            fileInput.value = ''; // Clear input
            
            dropArea.style.display = 'flex';
            selectedState.style.display = 'none';
            
            if (datasetStatus) datasetStatus.classList.remove('ready');
            if (datasetStatusText) datasetStatusText.textContent = 'Waiting for dataset';
        });
    }

    // Handle Form Submit Loading State
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const loadingIndicator = document.getElementById('loading-indicator');
            const uploadActions = document.getElementById('upload-actions');
            if (loadingIndicator && uploadActions) {
                uploadActions.style.display = 'none';
                loadingIndicator.style.display = 'flex';
            }
        });
    }
});

/**
 * Smart Paired Select Validation (Global System)
 * Prevents invalid dataset column combinations.
 * @param {HTMLElement|HTMLElement[]} primaries - Primary selector(s)
 * @param {HTMLElement|HTMLElement[]} secondaries - Secondary selector(s) or checkboxes
 * @param {HTMLButtonElement} actionButton - Button to disable
 * @param {Function} customValidation - Custom validation logic
 */
function setupPairedSelectValidation(primaries, secondaries, actionButton = null, customValidation = null) {
    if (!primaries || !secondaries) return;

    const primList = Array.isArray(primaries) ? primaries : [primaries];
    const secList = Array.isArray(secondaries) ? secondaries : (secondaries instanceof NodeList ? Array.from(secondaries) : [secondaries]);

    if (primList.length === 0 || secList.length === 0) return;

    function updateOptions() {
        const primaryVals = primList.map(p => p.value).filter(v => v !== "");
        let needsValidate = false;

        secList.forEach(sec => {
            if (sec.tagName === 'SELECT') {
                let needsReset = false;
                Array.from(sec.options).forEach(opt => {
                    if (opt.value === "" || (opt.disabled && !opt.hasAttribute('data-dynamic-disabled'))) {
                        return;
                    }
                    if (primaryVals.includes(opt.value)) {
                        opt.disabled = true;
                        opt.setAttribute('data-dynamic-disabled', 'true');
                        if (opt.selected) needsReset = true;
                    } else if (opt.hasAttribute('data-dynamic-disabled')) {
                        opt.disabled = false;
                        opt.removeAttribute('data-dynamic-disabled');
                    }
                });
                if (needsReset) sec.value = "";
            } else if (sec.type === 'checkbox') {
                if (primaryVals.includes(sec.value)) {
                    if (!sec.disabled) {
                        sec.disabled = true;
                        sec.setAttribute('data-dynamic-disabled', 'true');
                        sec.checked = false; // Prevent selection if invalid
                    }
                } else if (sec.hasAttribute('data-dynamic-disabled')) {
                    sec.disabled = false;
                    sec.removeAttribute('data-dynamic-disabled');
                }
            }
        });
        
        validateState();
    }

    function validateState() {
        if (!actionButton) return;
        
        let isValid = true;
        const primaryVals = primList.map(p => p.value).filter(v => v !== "");
        
        // Base overlap validation for selects
        secList.forEach(sec => {
            if (sec.tagName === 'SELECT' && sec.value && primaryVals.includes(sec.value)) {
                isValid = false;
            }
        });

        if (customValidation && typeof customValidation === 'function') {
            if (!customValidation()) {
                isValid = false;
            }
        }
        
        actionButton.disabled = !isValid;
    }

    primList.forEach(p => {
        p.addEventListener('change', updateOptions);
        p.triggerPairedUpdate = updateOptions;
    });
    
    secList.forEach(sec => {
        sec.addEventListener('change', validateState);
        sec.triggerPairedValidate = validateState;
    });
    
    updateOptions();
}

window.setupPairedSelectValidation = setupPairedSelectValidation;
