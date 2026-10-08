document.addEventListener('DOMContentLoaded', function() {
    const btnRunPca = document.getElementById('btn-run-pca');
    const varianceThreshold = document.getElementById('variance-threshold');
    const componentSelector = document.getElementById('component-selector');
    const colorBy = document.getElementById('color-by');
    
    let pcaResults = null;
    let featureScreening = null;
    
    // Load feature screening on page load
    loadFeatureScreening();
    
    if (btnRunPca) {
        btnRunPca.addEventListener('click', runPca);
    }
    
    if (componentSelector) {
        componentSelector.addEventListener('change', updateLoadingsTable);
    }
    
    if (colorBy) {
        colorBy.addEventListener('change', updateProjectionChart);
    }
    
    function loadFeatureScreening() {
        fetch('/pca/features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                featureScreening = data;
                updateFeatureSummary(data);
                updateFeatureScreeningTable(data.feature_screening);
            } else {
                console.error('Failed to load feature screening:', data.error);
            }
        })
        .catch(error => {
            console.error('Error loading feature screening:', error);
        });
    }
    
    function updateFeatureSummary(data) {
        document.getElementById('summary-dataset-cols').textContent = data.dataset_columns;
        document.getElementById('summary-target').textContent = data.target;
        document.getElementById('summary-modeling-feats').textContent = data.modeling_features_count;
        document.getElementById('summary-excluded-feats').textContent = data.excluded_features_count;
        document.getElementById('summary-num-feats').textContent = data.numerical_features_count;
        document.getElementById('summary-cat-feats').textContent = data.categorical_features_count;
    }
    
    function updateFeatureScreeningTable(screeningData) {
        const tbody = document.getElementById('feature-screening-tbody');
        tbody.innerHTML = '';
        
        // Show first 20 features for performance
        const displayData = screeningData.slice(0, 20);
        
        displayData.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="font-medium">${item.feature}</td>
                <td>${item.type}</td>
                <td><span class="badge ${item.role === 'Target' ? 'bg-warning-light text-warning-dark' : 'bg-success-light text-success-dark'}">${item.role}</span></td>
                <td>${item.used_for_pca ? '<i class="ph ph-check-circle text-success"></i> Yes' : '<i class="ph ph-x-circle text-danger"></i> No'}</td>
                <td class="text-sm text-muted">${item.reason}</td>
            `;
            tbody.appendChild(row);
        });
        
        if (screeningData.length > 20) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="5" class="text-center text-muted text-sm">... and ${screeningData.length - 20} more features</td>`;
            tbody.appendChild(row);
        }
    }
    
    function runPca() {
        btnRunPca.disabled = true;
        btnRunPca.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Running PCA...';
        
        const threshold = parseFloat(varianceThreshold.value);
        
        fetch('/pca/apply', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                variance_threshold: threshold
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Server error: ${response.status} - ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                pcaResults = data;
                displayPcaResults(data);
            } else {
                alert('PCA failed: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error running PCA:', error);
            alert('Error running PCA: ' + error.message);
        })
        .finally(() => {
            btnRunPca.disabled = false;
            btnRunPca.innerHTML = '<i class="ph ph-play"></i> Run PCA';
        });
    }
    
    function displayPcaResults(data) {
        // Show all sections
        document.getElementById('pca-summary-section').classList.remove('hidden');
        document.getElementById('explained-variance-section').classList.remove('hidden');
        document.getElementById('components-table-section').classList.remove('hidden');
        document.getElementById('loadings-section').classList.remove('hidden');
        document.getElementById('projection-section').classList.remove('hidden');
        document.getElementById('eli5-section').classList.remove('hidden');
        
        // Update summary cards
        document.getElementById('pca-orig-feats').textContent = data.summary.original_features;
        document.getElementById('pca-components').textContent = data.summary.selected_components;
        document.getElementById('pca-var-retained').textContent = data.summary.variance_retained + '%';
        document.getElementById('pca-var-lost').textContent = data.summary.variance_lost + '%';
        document.getElementById('pca-compression').textContent = data.summary.compression;
        
        // Update components table
        updateComponentsTable(data.components);
        
        // Populate component selector
        populateComponentSelector(data.components);
        
        // Update loadings table for first component
        updateLoadingsTable();
        
        // Draw charts
        drawExplainedVarianceChart(data.explained_variance);
        drawCumulativeVarianceChart(data.explained_variance);
        drawProjectionChart(data.projection_2d);
    }
    
    function updateComponentsTable(components) {
        const tbody = document.getElementById('components-tbody');
        tbody.innerHTML = '';
        
        components.forEach(comp => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="font-medium">${comp.name}</td>
                <td>${comp.explained_variance.toFixed(6)}</td>
                <td>${comp.explained_variance_ratio}%</td>
                <td>${comp.cumulative_variance}%</td>
            `;
            tbody.appendChild(row);
        });
    }
    
    function populateComponentSelector(components) {
        componentSelector.innerHTML = '';
        components.forEach((comp, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = comp.name;
            componentSelector.appendChild(option);
        });
    }
    
    function updateLoadingsTable() {
        if (!pcaResults) return;
        
        const componentIndex = parseInt(componentSelector.value);
        const loadings = pcaResults.loadings[componentIndex];
        
        const tbody = document.getElementById('loadings-tbody');
        tbody.innerHTML = '';
        
        // Show top 20 features by absolute loading
        const topLoadings = loadings.slice(0, 20);
        
        topLoadings.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="font-medium">${item.feature}</td>
                <td>${item.loading.toFixed(4)}</td>
                <td>${item.absolute_loading.toFixed(4)}</td>
                <td><span class="badge ${item.direction === 'Positive' ? 'bg-success-light text-success-dark' : 'bg-danger-light text-danger-dark'}">${item.direction}</span></td>
            `;
            tbody.appendChild(row);
        });
        
        if (loadings.length > 20) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="4" class="text-center text-muted text-sm">... and ${loadings.length - 20} more features</td>`;
            tbody.appendChild(row);
        }
    }
    
    function drawExplainedVarianceChart(varianceData) {
        const ctx = document.getElementById('explained-variance-chart').getContext('2d');
        
        if (window.explainedVarianceChart) {
            window.explainedVarianceChart.destroy();
        }
        
        const labels = varianceData.individual.map((_, i) => `PC${i + 1}`);
        
        window.explainedVarianceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Explained Variance Ratio (%)',
                    data: varianceData.individual,
                    backgroundColor: 'rgba(0, 86, 179, 0.6)',
                    borderColor: 'rgba(0, 86, 179, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Explained Variance Ratio (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Principal Component'
                        }
                    }
                }
            }
        });
    }
    
    function drawCumulativeVarianceChart(varianceData) {
        const ctx = document.getElementById('cumulative-variance-chart').getContext('2d');
        
        if (window.cumulativeVarianceChart) {
            window.cumulativeVarianceChart.destroy();
        }
        
        const labels = varianceData.cumulative.map((_, i) => `PC${i + 1}`);
        
        window.cumulativeVarianceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cumulative Explained Variance (%)',
                    data: varianceData.cumulative,
                    borderColor: 'rgba(0, 150, 136, 1)',
                    backgroundColor: 'rgba(0, 150, 136, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Cumulative Variance (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Principal Component'
                        }
                    }
                }
            }
        });
    }
    
    function drawProjectionChart(projectionData) {
        const ctx = document.getElementById('pca-projection-chart').getContext('2d');
        
        if (window.projectionChart) {
            window.projectionChart.destroy();
        }
        
        const colorByValue = colorBy.value;
        
        // Prepare data points
        const points = projectionData.map(p => ({
            x: p.pc1,
            y: p.pc2,
            target: p.target
        }));
        
        // Create datasets based on color selection
        let datasets = [];
        
        if (colorByValue === 'attrition') {
            // Separate by target (0 = No, 1 = Yes)
            const pointsNo = points.filter(p => p.target === 0);
            const pointsYes = points.filter(p => p.target === 1);
            
            datasets = [
                {
                    label: 'Attrition: No',
                    data: pointsNo,
                    backgroundColor: 'rgba(0, 150, 136, 0.6)',
                    borderColor: 'rgba(0, 150, 136, 1)',
                    pointRadius: 4
                },
                {
                    label: 'Attrition: Yes',
                    data: pointsYes,
                    backgroundColor: 'rgba(255, 112, 67, 0.6)',
                    borderColor: 'rgba(255, 112, 67, 1)',
                    pointRadius: 4
                }
            ];
        } else {
            // Single color
            datasets = [{
                label: 'Employees',
                data: points,
                backgroundColor: 'rgba(0, 86, 179, 0.6)',
                borderColor: 'rgba(0, 86, 179, 1)',
                pointRadius: 4
            }];
        }
        
        window.projectionChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `PC1: ${point.x.toFixed(2)}, PC2: ${point.y.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'PC1'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'PC2'
                        }
                    }
                }
            }
        });
    }
    
    function updateProjectionChart() {
        if (!pcaResults) return;
        drawProjectionChart(pcaResults.projection_2d);
    }
});
