document.addEventListener('DOMContentLoaded', function() {
    const modelSelector = document.getElementById('model-selector');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const compareBtn = document.getElementById('compare-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const evaluationResults = document.getElementById('evaluation-results');
    const classificationResults = document.getElementById('classification-results');
    const regressionResults = document.getElementById('regression-results');
    
    let rocChart = null;
    
    // Enable evaluate button when model is selected
    modelSelector.addEventListener('change', function() {
        evaluateBtn.disabled = !this.value;
    });
    
    // Evaluate single model
    evaluateBtn.addEventListener('click', async function() {
        const modelName = modelSelector.value;
        const modelType = modelSelector.options[modelSelector.selectedIndex].dataset.type;
        
        if (!modelName) return;
        
        showLoading();
        
        try {
            const response = await fetch('/evaluation/model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: modelName,
                    model_type: modelType
                })
            });
            
            const result = await response.json();
            
            hideLoading();
            
            if (result.success) {
                if (modelType === 'regression') {
                    displayRegressionResults(result);
                } else {
                    displayClassificationResults(result);
                }
                evaluationResults.style.display = 'block';
            } else {
                alert('Evaluation failed: ' + result.error);
            }
        } catch (error) {
            hideLoading();
            alert('Error: ' + error.message);
        }
    });
    
    // Compare all models
    compareBtn.addEventListener('click', async function() {
        showLoading();
        
        try {
            const response = await fetch('/evaluation/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            hideLoading();
            
            if (result.success) {
                displayComparisonResults(result.comparison);
                document.getElementById('comparison-results').style.display = 'block';
            } else {
                alert('Comparison failed: ' + result.error);
            }
        } catch (error) {
            hideLoading();
            alert('Error: ' + error.message);
        }
    });
    
    function displayClassificationResults(result) {
        classificationResults.style.display = 'block';
        regressionResults.style.display = 'none';
        
        // Update metrics
        document.getElementById('metric-accuracy').textContent = (result.metrics.accuracy * 100).toFixed(2) + '%';
        document.getElementById('metric-precision').textContent = (result.metrics.precision * 100).toFixed(2) + '%';
        document.getElementById('metric-recall').textContent = (result.metrics.recall * 100).toFixed(2) + '%';
        document.getElementById('metric-f1').textContent = (result.metrics.f1_score * 100).toFixed(2) + '%';
        
        // Update confusion matrix
        document.getElementById('cm-tn').textContent = result.confusion_matrix.true_negative;
        document.getElementById('cm-fp').textContent = result.confusion_matrix.false_positive;
        document.getElementById('cm-fn').textContent = result.confusion_matrix.false_negative;
        document.getElementById('cm-tp').textContent = result.confusion_matrix.true_positive;
        
        // Update classification report
        const reportBody = document.getElementById('classification-report-body');
        reportBody.innerHTML = '';
        
        result.classification_report.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.class}</td>
                <td>${(row.precision * 100).toFixed(2)}%</td>
                <td>${(row.recall * 100).toFixed(2)}%</td>
                <td>${(row.f1_score * 100).toFixed(2)}%</td>
                <td>${row.support}</td>
            `;
            reportBody.appendChild(tr);
        });
        
        // Update ROC curve if available
        if (result.roc_curve.fpr && result.roc_curve.tpr) {
            document.getElementById('roc-card').style.display = 'block';
            document.getElementById('roc-auc-value').textContent = result.metrics.roc_auc ? 
                (result.metrics.roc_auc * 100).toFixed(2) + '%' : 'N/A';
            
            renderROCChart(result.roc_curve.fpr, result.roc_curve.tpr, result.metrics.roc_auc);
        } else {
            document.getElementById('roc-card').style.display = 'none';
        }
    }
    
    function displayRegressionResults(result) {
        classificationResults.style.display = 'none';
        regressionResults.style.display = 'block';
        
        document.getElementById('regression-target').textContent = result.target;
        document.getElementById('metric-mae').textContent = result.metrics.mae.toFixed(4);
        document.getElementById('metric-mse').textContent = result.metrics.mse.toFixed(4);
        document.getElementById('metric-rmse').textContent = result.metrics.rmse.toFixed(4);
        document.getElementById('metric-r2').textContent = result.metrics.r2.toFixed(4);
    }
    
    function displayComparisonResults(comparison) {
        const tableBody = document.getElementById('comparison-table-body');
        tableBody.innerHTML = '';
        
        comparison.forEach(model => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${model.model}</td>
                <td>${(model.accuracy * 100).toFixed(2)}%</td>
                <td>${(model.precision * 100).toFixed(2)}%</td>
                <td>${(model.recall * 100).toFixed(2)}%</td>
                <td>${(model.f1_score * 100).toFixed(2)}%</td>
                <td>${model.roc_auc ? (model.roc_auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            `;
            tableBody.appendChild(tr);
        });
    }
    
    function renderROCChart(fpr, tpr, auc) {
        const ctx = document.getElementById('roc-chart').getContext('2d');
        
        if (rocChart) {
            rocChart.destroy();
        }
        
        rocChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Model ROC',
                    data: fpr.map((fp, i) => ({ x: fp, y: tpr[i] })),
                    borderColor: '#3AAFA9',
                    backgroundColor: 'rgba(58, 175, 169, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Random Baseline',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: '#999',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `ROC Curve (AUC = ${auc ? (auc * 100).toFixed(2) + '%' : 'N/A'})`
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `FPR: ${context.parsed.x.toFixed(3)}, TPR: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }
    
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
});
