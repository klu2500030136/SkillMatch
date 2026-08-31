document.addEventListener('DOMContentLoaded', () => {
    const analysisForm = document.getElementById('analysis-form');
    const hrQuestionForm = document.getElementById('hr-question-form');
    
    if (!analysisForm) return; // Exit if no dataset loaded
    
    const featureSelect = document.getElementById('feature-select');
    const methodSelect = document.getElementById('analysis-method');
    const groupColContainer = document.getElementById('group-col-container');
    const groupColSelect = document.getElementById('group-col-select');
    const btnRunAnalysis = document.getElementById('btn-run-analysis');
    const btnClearAnalysis = document.getElementById('btn-clear-analysis');
    const btnRunQuestion = document.getElementById('btn-run-question');
    const questionSelect = document.getElementById('question-select');
    
    const resultsContainer = document.getElementById('results-container');
    const resultTitle = document.getElementById('result-title');
    const resultSummary = document.getElementById('result-summary');
    const resultContent = document.getElementById('result-content');
    const resultError = document.getElementById('result-error');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    const chartContainer = document.getElementById('chart-container');
    const tableContainer = document.getElementById('table-container');
    const tableHead = document.getElementById('result-table-head');
    const tableBody = document.getElementById('result-table-body');
    
    let currentChart = null;

    // Apply smart paired validation if available
    if (window.setupPairedSelectValidation) {
        window.setupPairedSelectValidation(featureSelect, groupColSelect, btnRunAnalysis, () => {
            if (methodSelect.value === 'correlation') {
                const selectedOption = featureSelect.options[featureSelect.selectedIndex];
                const featureType = selectedOption ? selectedOption.getAttribute('data-type') : '';
                if (featureType !== 'Numerical') {
                    return false;
                }
            }
            // For group comparison we need both to be selected
            if (methodSelect.value === 'group_comparison') {
                if (!featureSelect.value || !groupColSelect.value) {
                    return false;
                }
            } else {
                if (!featureSelect.value && featureSelect.required) {
                    return false;
                }
            }
            return true;
        });
        
        // Ensure validation logic runs on method change too
        methodSelect.addEventListener('change', () => {
            if (groupColSelect.triggerPairedValidate) groupColSelect.triggerPairedValidate();
        });
    }

    // Toggle Group column visibility based on method
    methodSelect.addEventListener('change', () => {
        if (methodSelect.value === 'group_comparison') {
            groupColContainer.classList.remove('hidden');
        } else {
            groupColContainer.classList.add('hidden');
        }
    });

    // Clear analysis
    btnClearAnalysis.addEventListener('click', () => {
        featureSelect.value = '';
        methodSelect.value = '';
        questionSelect.value = '';
        groupColContainer.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }
    });

    // Run standard analysis
    analysisForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const feature = featureSelect.value;
        const method = methodSelect.value;
        const groupCol = groupColSelect.value;
        
        const selectedOption = featureSelect.options[featureSelect.selectedIndex];
        const featureType = selectedOption ? selectedOption.getAttribute('data-type') : '';
        
        if (method === 'correlation' && featureType !== 'Numerical') {
            showError("Correlation analysis requires a numerical feature.");
            return;
        }
        
        const payload = {
            analysis_type: method,
            feature: feature,
            group_col: method === 'group_comparison' ? groupCol : null
        };
        
        fetchAnalysis(payload);
    });

    // Run HR question
    hrQuestionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const payload = {
            question_id: questionSelect.value
        };
        fetchAnalysis(payload);
    });

    function fetchAnalysis(payload) {
        // Reset UI
        resultsContainer.classList.remove('hidden');
        resultContent.classList.add('hidden');
        resultError.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        
        // Disable buttons
        btnRunAnalysis.disabled = true;
        btnRunQuestion.disabled = true;

        fetch('/analyse/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            loadingSpinner.classList.add('hidden');
            btnRunAnalysis.disabled = false;
            btnRunQuestion.disabled = false;

            if (data.success) {
                renderResult(data);
            } else {
                showError(data.error || "An unknown error occurred.");
            }
        })
        .catch(err => {
            loadingSpinner.classList.add('hidden');
            btnRunAnalysis.disabled = false;
            btnRunQuestion.disabled = false;
            showError("A network error occurred while running the analysis.");
            console.error(err);
        });
    }

    function showError(message) {
        resultError.textContent = message;
        resultError.classList.remove('hidden');
        resultContent.classList.add('hidden');
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }
    }

    function renderResult(data) {
        resultContent.classList.remove('hidden');
        resultError.classList.add('hidden');
        
        // Title & Summary
        resultTitle.textContent = data.title || "Analysis Result";
        
        if (data.summary) {
            resultSummary.textContent = data.summary;
            resultSummary.classList.remove('hidden');
        } else {
            resultSummary.classList.add('hidden');
        }

        // Table
        if (data.table && data.table.columns && data.table.rows) {
            tableContainer.classList.remove('hidden');
            
            // Render Head
            let headHTML = '<tr>';
            data.table.columns.forEach(col => {
                headHTML += `<th>${col}</th>`;
            });
            headHTML += '</tr>';
            tableHead.innerHTML = headHTML;
            
            // Render Body
            let bodyHTML = '';
            data.table.rows.forEach(row => {
                bodyHTML += '<tr>';
                data.table.columns.forEach(col => {
                    let val = row[col];
                    if (val === null || val === undefined) val = '-';
                    bodyHTML += `<td>${val}</td>`;
                });
                bodyHTML += '</tr>';
            });
            tableBody.innerHTML = bodyHTML;
            
        } else {
            tableContainer.classList.add('hidden');
        }

        // Chart
        if (data.chart) {
            chartContainer.classList.remove('hidden');
            renderChart(data.chart);
        } else {
            chartContainer.classList.add('hidden');
            if (currentChart) {
                currentChart.destroy();
                currentChart = null;
            }
        }
    }

    function renderChart(chartConfig) {
        const ctx = document.getElementById('analysisChart').getContext('2d');
        
        if (currentChart) {
            currentChart.destroy();
        }
        
        // Add responsive config
        chartConfig.options = chartConfig.options || {};
        chartConfig.options.responsive = true;
        chartConfig.options.maintainAspectRatio = false;
        
        currentChart = new Chart(ctx, chartConfig);
    }
});
