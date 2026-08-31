document.addEventListener('DOMContentLoaded', function() {
    // Check if on statistics page with dataset loaded
    if (!document.getElementById('stat-filter')) return;

    // Filter and Search functionality
    const statFilter = document.getElementById('stat-filter');
    const statSearch = document.getElementById('stat-search');
    const numSection = document.getElementById('numerical-section');
    const catSection = document.getElementById('categorical-section');
    
    function applyFilters() {
        const filterVal = statFilter.value;
        const searchVal = statSearch.value.toLowerCase().trim();
        
        // Handle Section visibility
        if (filterVal === 'numerical') {
            numSection.style.display = 'block';
            catSection.style.display = 'none';
        } else if (filterVal === 'categorical') {
            numSection.style.display = 'none';
            catSection.style.display = 'block';
        } else {
            numSection.style.display = 'block';
            catSection.style.display = 'block';
        }
        
        // Handle Search row filtering
        const numRows = document.querySelectorAll('.num-row');
        let numVisible = 0;
        numRows.forEach(row => {
            const feature = row.getAttribute('data-feature');
            if (feature.includes(searchVal)) {
                row.style.display = '';
                numVisible++;
            } else {
                row.style.display = 'none';
            }
        });
        
        const catRows = document.querySelectorAll('.cat-row');
        let catVisible = 0;
        catRows.forEach(row => {
            const feature = row.getAttribute('data-feature');
            if (feature.includes(searchVal)) {
                row.style.display = '';
                catVisible++;
            } else {
                row.style.display = 'none';
            }
        });
    }
    
    statFilter.addEventListener('change', applyFilters);
    statSearch.addEventListener('input', applyFilters);

    // Inspector functionality
    const inspectSelect = document.getElementById('inspect-column');
    const inspectSpinner = document.getElementById('inspect-spinner');
    const inspectResult = document.getElementById('inspect-result');
    const inspectContent = document.getElementById('inspect-content');
    const inspectError = document.getElementById('inspect-error');
    
    const inspectTitle = document.getElementById('inspect-title');
    const inspectStatsBody = document.getElementById('inspect-stats-body');
    const chartTitle = document.getElementById('chart-title');
    const inspectorCatTable = document.getElementById('inspector-cat-table');
    const inspectorTableContainer = document.getElementById('inspector-table-container');
    
    let inspectorChartInstance = null;
    
    inspectSelect.addEventListener('change', function() {
        const column = this.value;
        if (!column) return;
        
        inspectSpinner.classList.remove('hidden');
        inspectResult.classList.remove('hidden');
        inspectContent.classList.add('hidden');
        inspectError.classList.add('hidden');
        
        fetch('/statistics/column', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ column: column })
        })
        .then(response => response.json())
        .then(data => {
            inspectSpinner.classList.add('hidden');
            
            if (data.error) {
                inspectError.textContent = data.error;
                inspectError.classList.remove('hidden');
                return;
            }
            
            inspectContent.classList.remove('hidden');
            
            // Render Stats
            const stats = data.stats;
            inspectTitle.textContent = `${stats.name} Statistics`;
            
            let statsHtml = `
                <tr><td class="font-medium text-muted">Data Type</td><td>${stats.dtype}</td></tr>
                <tr><td class="font-medium text-muted">Count</td><td>${stats.count}</td></tr>
                <tr><td class="font-medium text-muted">Missing</td><td>${stats.missing}</td></tr>
            `;
            
            if (stats.type === 'numerical') {
                statsHtml += `
                    <tr><td class="font-medium text-muted">Mean</td><td>${stats.mean !== null ? stats.mean : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Median</td><td>${stats.median !== null ? stats.median : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Std Dev</td><td>${stats.std_dev !== null ? stats.std_dev : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Min</td><td>${stats.min !== null ? stats.min : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Q1</td><td>${stats.q1 !== null ? stats.q1 : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Q2</td><td>${stats.q2 !== null ? stats.q2 : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Q3</td><td>${stats.q3 !== null ? stats.q3 : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">Max</td><td>${stats.max !== null ? stats.max : 'N/A'}</td></tr>
                    <tr><td class="font-medium text-muted">IQR</td><td>${stats.iqr !== null ? stats.iqr : 'N/A'}</td></tr>
                `;
                if (stats.skewness !== null) statsHtml += `<tr><td class="font-medium text-muted">Skewness</td><td>${stats.skewness}</td></tr>`;
                if (stats.kurtosis !== null) statsHtml += `<tr><td class="font-medium text-muted">Kurtosis</td><td>${stats.kurtosis}</td></tr>`;
            } else {
                statsHtml += `
                    <tr><td class="font-medium text-muted">Unique Values</td><td>${stats.unique_values}</td></tr>
                    <tr><td class="font-medium text-muted">Most Frequent</td><td>${stats.most_frequent}</td></tr>
                    <tr><td class="font-medium text-muted">Frequency</td><td>${stats.frequency}</td></tr>
                    <tr><td class="font-medium text-muted">Percentage</td><td>${stats.percentage}%</td></tr>
                `;
            }
            inspectStatsBody.innerHTML = statsHtml;
            
            // Render Chart
            if (inspectorChartInstance) {
                inspectorChartInstance.destroy();
            }
            
            const dist = data.distribution;
            chartTitle.textContent = dist.title;
            
            const ctx = document.getElementById('inspectorChart').getContext('2d');
            
            if (dist.type === 'numerical') {
                inspectorTableContainer.classList.add('hidden');
                inspectorChartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: dist.labels,
                        datasets: [{
                            label: 'Frequency',
                            data: dist.data,
                            backgroundColor: '#3b82f6',
                            borderWidth: 1,
                            borderColor: '#2563eb'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { categoryPercentage: 1.0, barPercentage: 1.0 },
                            y: { beginAtZero: true }
                        }
                    }
                });
            } else {
                inspectorTableContainer.classList.remove('hidden');
                inspectorChartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: dist.labels,
                        datasets: [{
                            label: 'Count',
                            data: dist.data,
                            backgroundColor: '#14b8a6',
                            borderWidth: 1,
                            borderColor: '#0d9488'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
                
                let catTableHtml = '';
                dist.table_data.forEach(row => {
                    catTableHtml += `<tr>
                        <td>${row.Category}</td>
                        <td>${row.Count}</td>
                        <td>${row.Percentage}%</td>
                    </tr>`;
                });
                inspectorCatTable.innerHTML = catTableHtml;
            }
            
        })
        .catch(error => {
            inspectSpinner.classList.add('hidden');
            inspectError.textContent = 'An error occurred while fetching column statistics.';
            inspectError.classList.remove('hidden');
            console.error(error);
        });
    });
});
