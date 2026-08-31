document.addEventListener('DOMContentLoaded', function() {
    if (!document.getElementById('preprocess-form')) return;

    const form = document.getElementById('preprocess-form');
    const targetDistInfo = document.getElementById('target-distribution-info');
    
    const splitInputs = document.querySelectorAll('.split-input');
    const splitError = document.getElementById('split-error');
    const btnApply = document.getElementById('btn-apply-preprocessing');
    const preprocessResults = document.getElementById('preprocessing-results');
    const preprocessConfigArea = document.getElementById('preprocess-config-area');
    const btnReconfigure = document.getElementById('btn-reconfigure');
    const btnViewResults = document.getElementById('btn-view-results');
    
    // Smart Validation: Target column should not be selectable as a removed feature
    if (window.setupPairedSelectValidation) {
        const targetColumnSelect = document.getElementById('target_column');
        const removeFeatureCheckboxes = document.querySelectorAll('.remove-feature-cb');
        if (targetColumnSelect && removeFeatureCheckboxes.length > 0) {
            window.setupPairedSelectValidation(
                targetColumnSelect, 
                removeFeatureCheckboxes, 
                btnApply,
                () => {
                    // Custom validation if needed
                    // Ensure split size is valid
                    const train = parseFloat(document.getElementById('train_size').value) || 0;
                    const val = parseFloat(document.getElementById('val_size').value) || 0;
                    const test = parseFloat(document.getElementById('test_size').value) || 0;
                    if (Math.abs((train + val + test) - 100) > 0.1) {
                        return false;
                    }
                    return true;
                }
            );
        }
    }

    // Check initial state
    const stateDataEl = document.getElementById('preprocessing-state-data');
    let savedState = null;
    if (stateDataEl && stateDataEl.textContent.trim() !== 'null') {
        try {
            savedState = JSON.parse(stateDataEl.textContent);
        } catch (e) {
            console.error('Failed to parse preprocessing state');
        }
    }
    
    // Invalidate state on any form change
    function invalidateState() {
        const alert = document.querySelector('.alert-success');
        if (alert) {
            alert.classList.remove('alert-success');
            alert.classList.add('alert-warning');
            alert.innerHTML = `
                <div class="flex items-center gap-2 font-bold mb-1">
                    <i class="ph ph-warning-circle text-lg"></i> Configuration Changed
                </div>
                <p class="text-sm">Preprocessing must be applied again before moving to Phase 7.</p>
            `;
        }
    }
    
    form.addEventListener('input', invalidateState);
    form.addEventListener('change', invalidateState);
    
    // Validate split sum
    function validateSplits() {
        const train = parseFloat(document.getElementById('train_size').value) || 0;
        const val = parseFloat(document.getElementById('val_size').value) || 0;
        const test = parseFloat(document.getElementById('test_size').value) || 0;
        
        if (train + val + test !== 100) {
            splitError.classList.remove('hidden');
            btnApply.disabled = true;
            return false;
        } else {
            splitError.classList.add('hidden');
            btnApply.disabled = false;
            return true;
        }
    }
    
    splitInputs.forEach(input => input.addEventListener('input', validateSplits));
    
    // Form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!validateSplits()) return;
        
        const btnOriginalText = btnApply.innerHTML;
        btnApply.innerHTML = '<i class="ph ph-spinner animate-spin"></i> Processing...';
        btnApply.disabled = true;
        
        // Gather features to remove
        const featuresToRemove = [];
        document.querySelectorAll('.remove-feature-cb').forEach(cb => {
            if (cb.checked) {
                featuresToRemove.push(cb.value);
            }
        });
        
        const payload = {
            target_type: 'classification',
            target_column: 'Attrition',
            remove_duplicates: document.getElementById('remove_duplicates').checked,
            features_to_remove: featuresToRemove,
            num_imputation: document.getElementById('num_imputation').value,
            cat_imputation: document.getElementById('cat_imputation').value,
            scaling: document.getElementById('scaling').value,
            train_size: parseFloat(document.getElementById('train_size').value),
            val_size: parseFloat(document.getElementById('val_size').value),
            test_size: parseFloat(document.getElementById('test_size').value),
            random_state: parseInt(document.getElementById('random_state').value, 10)
        };
        
        fetch('/preprocessing/apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            btnApply.innerHTML = btnOriginalText;
            btnApply.disabled = false;
            
            if (data.error) {
                alert('Preprocessing failed: ' + data.error);
                return;
            }
            
            // Render results
            renderResults(data);
            
            // Show alert
            const alertContainer = document.querySelector('.alert');
            if (alertContainer) alertContainer.remove(); // clear old
            
            const newAlert = document.createElement('div');
            newAlert.className = 'alert alert-success mb-6';
            newAlert.innerHTML = `
                <div class="flex items-center gap-2 font-bold mb-1">
                    <i class="ph ph-check-circle text-lg"></i> Preprocessing Applied
                </div>
                <p class="text-sm">This dataset has been successfully preprocessed and is ready for Phase 7 (Model Training).</p>
            `;
            preprocessConfigArea.parentNode.insertBefore(newAlert, preprocessConfigArea);
            
            // Switch view
            preprocessConfigArea.classList.add('hidden');
            preprocessResults.classList.remove('hidden');
        })
        .catch(err => {
            console.error(err);
            btnApply.innerHTML = btnOriginalText;
            btnApply.disabled = false;
            alert('An unexpected error occurred.');
        });
    });
    
    function renderResults(data) {
        document.getElementById('res-orig-rows').textContent = data.summary.original_rows;
        document.getElementById('res-orig-cols').textContent = data.summary.original_cols;
        
        document.getElementById('res-proc-rows').textContent = data.summary.processed_rows;
        document.getElementById('res-proc-feats').textContent = data.summary.processed_features;
        document.getElementById('res-dups-removed').textContent = data.summary.duplicates_removed;
        document.getElementById('res-feats-removed').textContent = data.summary.features_removed;
        
        document.getElementById('res-train-rows').textContent = data.summary.train_rows;
        document.getElementById('res-val-rows').textContent = data.summary.val_rows;
        document.getElementById('res-test-rows').textContent = data.summary.test_rows;
        
        // Render Table Preview
        const thead = document.getElementById('preview-thead');
        const tbody = document.getElementById('preview-tbody');
        
        let theadHtml = '<tr><th>Target (' + data.summary.target + ')</th>';
        data.preview_columns.forEach(col => {
            theadHtml += `<th>${col}</th>`;
        });
        theadHtml += '</tr>';
        thead.innerHTML = theadHtml;
        
        let tbodyHtml = '';
        data.preview_data.forEach((row, i) => {
            const targetVal = data.target_preview[i];
            tbodyHtml += `<tr><td class="font-bold bg-primary-light text-primary-dark border-r border-primary">${targetVal}</td>`;
            data.preview_columns.forEach(col => {
                tbodyHtml += `<td>${row[col]}</td>`;
            });
            tbodyHtml += '</tr>';
        });
        tbody.innerHTML = tbodyHtml;
    }
    
    btnReconfigure.addEventListener('click', function() {
        preprocessResults.classList.add('hidden');
        preprocessConfigArea.classList.remove('hidden');
    });
    
    if (btnViewResults) {
        btnViewResults.addEventListener('click', function() {
            // Need to fetch or just show that it was processed? 
            // We don't store the preview in session to save space, but we can just say to re-apply if they want to see the preview.
            alert('To view the detailed preview again, please click "Apply Preprocessing" to re-generate the data splits.');
            preprocessConfigArea.classList.remove('hidden');
            preprocessResults.classList.add('hidden');
        });
    }
});
