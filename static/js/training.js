document.addEventListener('DOMContentLoaded', function() {
    if (!document.getElementById('training-form')) return;
    
    const form = document.getElementById('training-form');
    const modelCbs = document.querySelectorAll('.model-cb');
    const btnSelectAll = document.getElementById('btn-select-all');
    const btnTrain = document.getElementById('btn-train');
    const resultsArea = document.getElementById('training-results');
    const btnReset = document.getElementById('btn-reset-settings');
    const staleAlert = document.getElementById('stale-alert');
    let f1ChartInstance = null;
    
    // Select All
    btnSelectAll.addEventListener('click', () => {
        const allChecked = Array.from(modelCbs).every(cb => cb.checked);
        modelCbs.forEach(cb => {
            cb.checked = !allChecked;
            toggleConfigSection(cb);
        });
        btnSelectAll.textContent = allChecked ? 'Select All' : 'Deselect All';
    });
    
    // Toggle Config sections
    function toggleConfigSection(cb) {
        let sectionId;
        if (cb.value === 'Logistic Regression') sectionId = 'cfg-lr';
        else if (cb.value === 'Decision Tree') sectionId = 'cfg-dt';
        else if (cb.value === 'Random Forest') sectionId = 'cfg-rf';
        else if (cb.value === 'K-Nearest Neighbors') sectionId = 'cfg-knn';
        else if (cb.value === 'Support Vector Machine') sectionId = 'cfg-svm';
        else if (cb.value === 'Gradient Boosting') sectionId = 'cfg-gb';
        
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = cb.checked ? 'block' : 'none';
        }
    }
    
    modelCbs.forEach(cb => {
        cb.addEventListener('change', () => toggleConfigSection(cb));
        toggleConfigSection(cb); // init
    });
    
    // Reset Settings
    btnReset.addEventListener('click', () => {
        form.reset();
        modelCbs.forEach(cb => toggleConfigSection(cb));
    });
    
    // Form Submit
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedModels = Array.from(modelCbs).filter(cb => cb.checked).map(cb => cb.value);
        
        if (selectedModels.length === 0) {
            alert('Please select at least one model to train.');
            return;
        }
        
        // Gather config
        const config = {
            lr_C: document.getElementById('lr_C').value,
            lr_max_iter: document.getElementById('lr_max_iter').value,
            dt_max_depth: document.getElementById('dt_max_depth').value,
            dt_min_samples_split: document.getElementById('dt_min_samples_split').value,
            dt_criterion: document.getElementById('dt_criterion').value,
            rf_n_estimators: document.getElementById('rf_n_estimators').value,
            rf_max_depth: document.getElementById('rf_max_depth').value,
            rf_min_samples_split: document.getElementById('rf_min_samples_split').value,
            knn_n_neighbors: document.getElementById('knn_n_neighbors').value,
            knn_weights: document.getElementById('knn_weights').value,
            svm_C: document.getElementById('svm_C').value,
            svm_kernel: document.getElementById('svm_kernel').value,
            gb_n_estimators: document.getElementById('gb_n_estimators').value,
            gb_learning_rate: document.getElementById('gb_learning_rate').value,
            gb_max_depth: document.getElementById('gb_max_depth').value
        };
        
        const payload = {
            selected_models: selectedModels,
            config: config,
            class_weight: document.getElementById('class_weight').value,
            random_state: document.getElementById('random_state').value
        };
        
        const originalText = btnTrain.innerHTML;
        btnTrain.innerHTML = '<i class="ph ph-spinner animate-spin"></i> Training...';
        btnTrain.disabled = true;
        
        fetch('/training/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            btnTrain.innerHTML = originalText;
            btnTrain.disabled = false;
            
            if (data.error) {
                alert('Training Error: ' + data.error);
                return;
            }
            
            if (staleAlert) staleAlert.style.display = 'none';
            renderResults(data);
        })
        .catch(err => {
            console.error(err);
            btnTrain.innerHTML = originalText;
            btnTrain.disabled = false;
            alert('An unexpected error occurred during training.');
        });
    });
    
    function renderResults(data) {
        resultsArea.classList.remove('hidden');
        
        const tbody = document.getElementById('results-tbody');
        tbody.innerHTML = '';
        
        const chartLabels = [];
        const chartData = [];
        const chartColors = [];
        
        // Sort by F1 descending
        const sortedModels = [...data.models].sort((a, b) => (b.f1 || 0) - (a.f1 || 0));
        
        sortedModels.forEach(m => {
            if (m.status === 'success') {
                const isBest = data.best_model && m.name === data.best_model.name;
                const rowClass = isBest ? 'bg-primary-light font-bold text-primary-dark' : '';
                
                tbody.innerHTML += `
                    <tr class="${rowClass}">
                        <td>${m.name} ${isBest ? '<i class="ph ph-star text-warning"></i>' : ''}</td>
                        <td>${m.accuracy}%</td>
                        <td>${m.precision}%</td>
                        <td>${m.recall}%</td>
                        <td>${m.f1}%</td>
                        <td>${m.roc_auc !== 'N/A' ? m.roc_auc + '%' : 'N/A'}</td>
                        <td>${m.training_time}</td>
                    </tr>
                `;
                
                chartLabels.push(m.name);
                chartData.push(m.f1);
                chartColors.push(isBest ? '#3b82f6' : '#94a3b8');
            } else {
                tbody.innerHTML += `
                    <tr class="text-danger">
                        <td>${m.name}</td>
                        <td colspan="6">Failed: ${m.reason}</td>
                    </tr>
                `;
            }
        });
        
        if (data.best_model) {
            document.getElementById('best-model-card').classList.remove('hidden');
            document.getElementById('bm-name').textContent = data.best_model.name;
            document.getElementById('bm-score').textContent = data.best_model.score + '%';
        } else {
            document.getElementById('best-model-card').classList.add('hidden');
        }
        
        renderChart(chartLabels, chartData, chartColors);
    }
    
    function renderChart(labels, data, colors) {
        const ctx = document.getElementById('f1Chart').getContext('2d');
        
        if (f1ChartInstance) {
            f1ChartInstance.destroy();
        }
        
        f1ChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Validation F1-score (%)',
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'F1-score (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
});
