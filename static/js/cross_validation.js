document.addEventListener('DOMContentLoaded', function() {
    const modelSelector = document.getElementById('model-selector');
    const runCvBtn = document.getElementById('run-cv-btn');
    const compareCvBtn = document.getElementById('compare-cv-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const cvResults = document.getElementById('cv-results');
    
    // Enable run button when model is selected
    modelSelector.addEventListener('change', function() {
        runCvBtn.disabled = !this.value;
    });
    
    // Run cross-validation for selected model
    runCvBtn.addEventListener('click', async function() {
        const modelName = modelSelector.value;
        
        if (!modelName) return;
        
        showLoading();
        
        try {
            const response = await fetch('/cross-validation/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: modelName,
                    n_splits: 5,
                    random_state: 42
                })
            });
            
            const result = await response.json();
            
            hideLoading();
            
            if (result.success) {
                displayCvResults(result);
                cvResults.style.display = 'block';
            } else {
                alert('Cross-validation failed: ' + result.error);
            }
        } catch (error) {
            hideLoading();
            alert('Error: ' + error.message);
        }
    });
    
    // Compare all models
    compareCvBtn.addEventListener('click', async function() {
        showLoading();
        
        try {
            const response = await fetch('/cross-validation/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            hideLoading();
            
            if (result.success) {
                displayCvComparison(result.comparison);
                document.getElementById('cv-comparison-results').style.display = 'block';
            } else {
                alert('Comparison failed: ' + result.error);
            }
        } catch (error) {
            hideLoading();
            alert('Error: ' + error.message);
        }
    });
    
    function displayCvResults(result) {
        // Display fold results
        const foldBody = document.getElementById('fold-results-body');
        foldBody.innerHTML = '';
        
        result.fold_results.forEach(fold => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${fold.fold}</td>
                <td>${(fold.accuracy * 100).toFixed(2)}%</td>
                <td>${(fold.precision * 100).toFixed(2)}%</td>
                <td>${(fold.recall * 100).toFixed(2)}%</td>
                <td>${(fold.f1 * 100).toFixed(2)}%</td>
                <td>${fold.roc_auc ? (fold.roc_auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            `;
            foldBody.appendChild(tr);
        });
        
        // Display summary
        const summaryBody = document.getElementById('summary-body');
        summaryBody.innerHTML = '';
        
        const summary = result.summary;
        
        // Mean row
        const meanRow = document.createElement('tr');
        meanRow.innerHTML = `
            <td><strong>Mean</strong></td>
            <td>${(summary.accuracy.mean * 100).toFixed(2)}%</td>
            <td>${(summary.precision.mean * 100).toFixed(2)}%</td>
            <td>${(summary.recall.mean * 100).toFixed(2)}%</td>
            <td>${(summary.f1.mean * 100).toFixed(2)}%</td>
            <td>${summary.roc_auc ? (summary.roc_auc.mean * 100).toFixed(2) + '%' : 'N/A'}</td>
        `;
        summaryBody.appendChild(meanRow);
        
        // Std Dev row
        const stdRow = document.createElement('tr');
        stdRow.innerHTML = `
            <td><strong>Std Dev</strong></td>
            <td>${(summary.accuracy.std * 100).toFixed(2)}%</td>
            <td>${(summary.precision.std * 100).toFixed(2)}%</td>
            <td>${(summary.recall.std * 100).toFixed(2)}%</td>
            <td>${(summary.f1.std * 100).toFixed(2)}%</td>
            <td>${summary.roc_auc ? (summary.roc_auc.std * 100).toFixed(2) + '%' : 'N/A'}</td>
        `;
        summaryBody.appendChild(stdRow);
    }
    
    function displayCvComparison(comparison) {
        const comparisonBody = document.getElementById('cv-comparison-body');
        comparisonBody.innerHTML = '';
        
        comparison.forEach(model => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${model.model}</td>
                <td>${(model.mean_accuracy * 100).toFixed(2)}%</td>
                <td>${(model.mean_precision * 100).toFixed(2)}%</td>
                <td>${(model.mean_recall * 100).toFixed(2)}%</td>
                <td>${(model.mean_f1 * 100).toFixed(2)}%</td>
                <td>${model.mean_roc_auc ? (model.mean_roc_auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            `;
            comparisonBody.appendChild(tr);
        });
    }
    
    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }
    
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
});
