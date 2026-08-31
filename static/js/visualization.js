document.addEventListener('DOMContentLoaded', function() {
    if (!document.getElementById('viz-form')) return;

    const chartType = document.getElementById('chart-type');
    const xAxisContainer = document.getElementById('x-axis-container');
    const yAxisContainer = document.getElementById('y-axis-container');
    const groupContainer = document.getElementById('group-container');
    const aggContainer = document.getElementById('agg-container');
    
    const xAxis = document.getElementById('x-axis');
    const yAxis = document.getElementById('y-axis');
    const groupAxis = document.getElementById('group-axis');
    const aggregation = document.getElementById('aggregation');
    
    const filterColumn = document.getElementById('filter-column');
    const filterValue = document.getElementById('filter-value');
    const filterOptionsData = JSON.parse(document.getElementById('filter-options-data').textContent || '{}');
    
    const vizForm = document.getElementById('viz-form');
    const btnClear = document.getElementById('btn-clear');
    
    const vizSpinner = document.getElementById('viz-spinner');
    const vizEmptyState = document.getElementById('viz-empty-state');
    const vizError = document.getElementById('viz-error');
    const vizContent = document.getElementById('viz-content');
    
    const vizTitle = document.getElementById('viz-title');
    const vizInsight = document.getElementById('viz-insight');
    const chartContainer = document.getElementById('chart-container');
    const heatmapContainer = document.getElementById('heatmap-container');
    const heatmapTable = document.getElementById('heatmap-table');
    const vizTable = document.getElementById('viz-table');
    const tableContainer = document.getElementById('table-container');
    
    let currentChart = null;
    const btnGenerate = document.getElementById('btn-generate');

    // Smart Paired Select Validation for Data Visualization
    if (window.setupPairedSelectValidation) {
        // X paired with both Y and Group to avoid conflicts
        window.setupPairedSelectValidation(xAxis, [yAxis, groupAxis], btnGenerate, () => {
            const type = chartType.value;
            const xVal = xAxis.value;
            
            if (!type) return false;
            
            if (type === 'heatmap') return true; // Heatmap requires no axes
            
            if (!xVal) return false;
            
            const isNum = xAxis.options[xAxis.selectedIndex]?.getAttribute('data-num') === 'true';

            if (type === 'histogram' && !isNum) return false;
            
            if (type === 'scatter') {
                const yVal = yAxis.value;
                if (!yVal) return false;
                const yIsNum = yAxis.options[yAxis.selectedIndex]?.getAttribute('data-num') === 'true';
                if (!isNum || !yIsNum) return false;
            }

            if (type === 'grouped_bar' || type === 'boxplot') {
                const gVal = groupAxis.value;
                if (!gVal) return false; // Both grouped bar and boxplot require a grouping column in this UI
            }

            return true;
        });
    }

    // Handle dynamic controls
    function updateControls() {
        const type = chartType.value;
        
        // Reset visibility
        xAxisContainer.classList.add('hidden');
        yAxisContainer.classList.add('hidden');
        groupContainer.classList.add('hidden');
        aggContainer.classList.add('hidden');
        
        xAxis.required = false;
        yAxis.required = false;
        groupAxis.required = false;
        
        if (type === 'histogram' || type === 'bar') {
            xAxisContainer.classList.remove('hidden');
            xAxis.required = true;
        } else if (type === 'scatter') {
            xAxisContainer.classList.remove('hidden');
            yAxisContainer.classList.remove('hidden');
            xAxis.required = true;
            yAxis.required = true;
        } else if (type === 'grouped_bar' || type === 'boxplot') {
            xAxisContainer.classList.remove('hidden');
            groupContainer.classList.remove('hidden');
            xAxis.required = true;
            groupAxis.required = true;
            
            if (type === 'grouped_bar') {
                aggContainer.classList.remove('hidden');
            }
        } else if (type === 'heatmap') {
            // No axes needed for correlation heatmap
        }
        
        if (xAxis.triggerPairedValidate) xAxis.triggerPairedValidate();
    }
    
    chartType.addEventListener('change', updateControls);
    
    // Handle dynamic filters
    filterColumn.addEventListener('change', function() {
        const col = this.value;
        if (!col) {
            filterValue.classList.add('hidden');
            filterValue.value = '';
            return;
        }
        
        filterValue.innerHTML = '<option value="" disabled selected>Select value...</option>';
        if (filterOptionsData[col]) {
            filterOptionsData[col].forEach(val => {
                const opt = document.createElement('option');
                opt.value = val;
                opt.textContent = val;
                filterValue.appendChild(opt);
            });
            filterValue.classList.remove('hidden');
        }
    });
    
    // Clear visualization
    function clearVisualization() {
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }
        
        vizEmptyState.classList.remove('hidden');
        vizContent.classList.add('hidden');
        vizError.classList.add('hidden');
        tableContainer.classList.add('hidden');
        heatmapContainer.classList.add('hidden');
        
        // Optional: clear form
        // vizForm.reset();
        // updateControls();
        // filterValue.classList.add('hidden');
    }
    
    btnClear.addEventListener('click', clearVisualization);
    
    // Helper to render heatmap
    function renderHeatmap(data) {
        let html = '<thead><tr><th></th>';
        data.labels.forEach(l => { html += `<th>${l}</th>`; });
        html += '</tr></thead><tbody>';
        
        for (let i = 0; i < data.labels.length; i++) {
            html += `<tr><th class="text-right pr-2">${data.labels[i]}</th>`;
            for (let j = 0; j < data.labels.length; j++) {
                const item = data.data.find(d => d.x === data.labels[j] && d.y === data.labels[i]);
                const val = item ? parseFloat(item.v).toFixed(2) : '0.00';
                
                // Color scaling (red = negative, blue = positive)
                let bgColor = '#ffffff';
                let textColor = '#000000';
                
                if (item && !isNaN(item.v)) {
                    const absVal = Math.abs(item.v);
                    if (item.v > 0) {
                        bgColor = `rgba(59, 130, 246, ${absVal})`; // blue
                        if (absVal > 0.5) textColor = '#ffffff';
                    } else if (item.v < 0) {
                        bgColor = `rgba(239, 68, 68, ${absVal})`; // red
                        if (absVal > 0.5) textColor = '#ffffff';
                    }
                }
                
                html += `<td style="background-color: ${bgColor}; color: ${textColor}; padding: 8px;">${val}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody>';
        heatmapTable.innerHTML = html;
        heatmapContainer.classList.remove('hidden');
        document.getElementById('vizChart').style.display = 'none';
    }
    
    // Form submission
    vizForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const type = chartType.value;
        const x = xAxis.value;
        const isNum = xAxis.options[xAxis.selectedIndex]?.getAttribute('data-num') === 'true';
        
        // Client-side validation
        if (type === 'histogram' && !isNum) {
            showError("Histogram requires a numerical feature. Please select a numerical X-Axis.");
            return;
        }
        if (type === 'scatter') {
            const yIsNum = yAxis.options[yAxis.selectedIndex]?.getAttribute('data-num') === 'true';
            if (!isNum || !yIsNum) {
                showError("Scatter plot requires numerical features for both X and Y axes.");
                return;
            }
        }
        
        const payload = {
            chart_type: type,
            x_column: x,
            y_column: yAxis.value,
            group_column: groupAxis.value,
            aggregation: aggregation.value,
            filter_column: filterColumn.value,
            filter_value: filterValue.value
        };
        
        generateChart(payload);
    });
    
    // Generate Chart Logic
    function generateChart(payload) {
        vizEmptyState.classList.add('hidden');
        vizError.classList.add('hidden');
        vizContent.classList.add('hidden');
        vizSpinner.classList.remove('hidden');
        
        fetch('/visualization/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(res => {
            vizSpinner.classList.add('hidden');
            if (res.error) {
                showError(res.error);
                return;
            }
            
            const data = res.data;
            vizTitle.textContent = data.title;
            vizInsight.textContent = data.insight;
            vizContent.classList.remove('hidden');
            
            if (currentChart) {
                currentChart.destroy();
                currentChart = null;
            }
            
            heatmapContainer.classList.add('hidden');
            const canvas = document.getElementById('vizChart');
            canvas.style.display = 'block';
            tableContainer.classList.add('hidden');
            
            if (data.chart_type === 'heatmap') {
                renderHeatmap(data);
                return;
            }
            
            // Standard Chart.js render
            const ctx = canvas.getContext('2d');
            
            const config = {
                type: data.chart_type,
                data: {
                    labels: data.labels,
                    datasets: data.datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: data.datasets.length > 1 }
                    }
                }
            };
            
            // Axes configuration
            if (data.chart_type !== 'pie' && data.chart_type !== 'doughnut') {
                config.options.scales = {
                    x: {
                        title: { display: !!data.x_label, text: data.x_label }
                    },
                    y: {
                        title: { display: !!data.y_label, text: data.y_label },
                        beginAtZero: true
                    }
                };
                
                if (data.stacked) {
                    config.options.scales.x.stacked = true;
                    config.options.scales.y.stacked = true;
                }
                
                if (data.chart_type === 'scatter') {
                    config.options.scales.x.type = 'linear';
                    config.options.scales.x.position = 'bottom';
                }
            }
            
            currentChart = new Chart(ctx, config);
            
            // If it's a grouped bar, also show a table
            if (data.chart_type === 'bar' && data.labels.length > 0 && !data.stacked && payload.group_column) {
                let html = `<thead><tr><th>${data.x_label}</th><th class="text-right">${data.y_label}</th></tr></thead><tbody>`;
                for (let i = 0; i < data.labels.length; i++) {
                    html += `<tr><td>${data.labels[i]}</td><td class="text-right">${data.datasets[0].data[i]}</td></tr>`;
                }
                html += `</tbody>`;
                vizTable.innerHTML = html;
                tableContainer.classList.remove('hidden');
            }
        })
        .catch(err => {
            vizSpinner.classList.add('hidden');
            showError("An unexpected error occurred while generating the visualization.");
            console.error(err);
        });
    }
    
    function showError(msg) {
        vizError.textContent = msg;
        vizError.classList.remove('hidden');
        vizEmptyState.classList.add('hidden');
        vizContent.classList.add('hidden');
    }

    // HR Quick Views functionality
    document.querySelectorAll('.hr-shortcut').forEach(btn => {
        btn.addEventListener('click', function() {
            const type = this.getAttribute('data-type');
            const x = this.getAttribute('data-x');
            const group = this.getAttribute('data-group');
            const agg = this.getAttribute('data-agg');
            
            // Update UI to match
            chartType.value = type;
            updateControls();
            
            xAxis.value = x;
            if (group) groupAxis.value = group;
            if (agg) aggregation.value = agg;
            
            // Reset filter
            filterColumn.value = '';
            filterValue.classList.add('hidden');
            filterValue.value = '';
            
            // Trigger form submission payload
            const payload = {
                chart_type: type,
                x_column: x,
                group_column: group,
                aggregation: agg,
                filter_column: null,
                filter_value: null
            };
            
            generateChart(payload);
        });
    });
});
