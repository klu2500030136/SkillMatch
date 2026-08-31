document.addEventListener('DOMContentLoaded', function() {
    // Check if on missing values page
    if (!document.getElementById('missing-filter')) return;

    // Filter and Search functionality
    const missingFilter = document.getElementById('missing-filter');
    const missingSearch = document.getElementById('missing-search');
    const tableBody = document.getElementById('missing-table-body');
    const noResultsMsg = document.getElementById('no-results-msg');
    const rows = Array.from(document.querySelectorAll('.missing-row'));
    
    function applyFilters() {
        const filterVal = missingFilter.value;
        const searchVal = missingSearch.value.toLowerCase().trim();
        let visibleCount = 0;
        
        rows.forEach(row => {
            const feature = row.getAttribute('data-feature');
            const status = row.getAttribute('data-status');
            const missingCount = parseInt(row.getAttribute('data-missing-count'), 10);
            
            let showStatus = false;
            if (filterVal === 'all') showStatus = true;
            else if (filterVal === 'affected' && missingCount > 0) showStatus = true;
            else if (filterVal === status) showStatus = true;
            
            const showSearch = feature.includes(searchVal);
            
            if (showStatus && showSearch) {
                row.style.display = '';
                visibleCount++;
            } else {
                row.style.display = 'none';
            }
        });
        
        if (visibleCount === 0 && rows.length > 0) {
            noResultsMsg.classList.remove('hidden');
        } else {
            noResultsMsg.classList.add('hidden');
        }
    }
    
    missingFilter.addEventListener('change', applyFilters);
    missingSearch.addEventListener('input', applyFilters);

    // Sorting functionality
    const headers = document.querySelectorAll('th.sortable');
    let currentSort = { column: 'missing_pct', direction: 'desc' };
    
    function sortTable(column) {
        if (currentSort.column === column) {
            currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
            currentSort.column = column;
            currentSort.direction = column === 'feature' || column === 'status' ? 'asc' : 'desc';
        }
        
        // Update header icons
        headers.forEach(th => {
            const icon = th.querySelector('i');
            if (th.getAttribute('data-sort') === currentSort.column) {
                icon.classList.remove('hidden');
                icon.className = currentSort.direction === 'asc' ? 'ph ph-caret-up' : 'ph ph-caret-down';
            } else {
                icon.classList.add('hidden');
            }
        });
        
        rows.sort((a, b) => {
            let valA, valB;
            
            if (currentSort.column === 'feature') {
                valA = a.getAttribute('data-feature');
                valB = b.getAttribute('data-feature');
            } else if (currentSort.column === 'status') {
                valA = a.getAttribute('data-status');
                valB = b.getAttribute('data-status');
            } else {
                // numeric sorting
                valA = parseFloat(a.getAttribute(`data-${currentSort.column.replace('_', '-')}`));
                valB = parseFloat(b.getAttribute(`data-${currentSort.column.replace('_', '-')}`));
            }
            
            if (valA < valB) return currentSort.direction === 'asc' ? -1 : 1;
            if (valA > valB) return currentSort.direction === 'asc' ? 1 : -1;
            return 0;
        });
        
        rows.forEach(row => tableBody.appendChild(row));
    }
    
    headers.forEach(th => {
        th.addEventListener('click', () => {
            sortTable(th.getAttribute('data-sort'));
        });
    });
    
    // Initial sort
    if (rows.length > 0) {
        sortTable('missing_pct'); // defaults to desc
    }

    // Chart.js visualizations
    const chartDataEl = document.getElementById('chart-data');
    if (chartDataEl) {
        const chartData = JSON.parse(chartDataEl.textContent);
        if (chartData.has_missing) {
            const ctxCount = document.getElementById('missingCountChart').getContext('2d');
            new Chart(ctxCount, {
                type: 'bar',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: 'Missing Count',
                        data: chartData.counts,
                        backgroundColor: '#f59e0b',
                        borderColor: '#d97706',
                        borderWidth: 1
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
            
            const ctxPct = document.getElementById('missingPctChart').getContext('2d');
            new Chart(ctxPct, {
                type: 'bar',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: 'Missing Percentage (%)',
                        data: chartData.percentages,
                        backgroundColor: '#ef4444',
                        borderColor: '#dc2626',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { 
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    }

    // Inspector functionality
    const inspectSelect = document.getElementById('inspect-feature');
    if (inspectSelect) {
        const inspectSpinner = document.getElementById('inspect-spinner');
        const inspectResult = document.getElementById('inspect-result');
        const inspectContent = document.getElementById('inspect-content');
        const inspectError = document.getElementById('inspect-error');
        const inspectTitle = document.getElementById('inspect-title');
        const inspectStatsBody = document.getElementById('inspect-stats-body');
        
        inspectSelect.addEventListener('change', function() {
            const feature = this.value;
            if (!feature) return;
            
            inspectSpinner.classList.remove('hidden');
            inspectResult.classList.remove('hidden');
            inspectContent.classList.add('hidden');
            inspectError.classList.add('hidden');
            
            fetch('/missing-values/feature', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ feature: feature })
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
                
                const stats = data.details;
                inspectTitle.textContent = `${stats.feature}`;
                
                let badgeClass = 'bg-success-light text-success-dark';
                if (stats.status === 'Low Missingness') badgeClass = 'bg-primary-light text-primary-dark';
                else if (stats.status === 'Moderate Missingness') badgeClass = 'bg-warning-light text-warning-dark';
                else if (stats.status === 'High Missingness') badgeClass = 'bg-danger-light text-danger-dark';
                
                if (stats.status === 'Moderate Missingness') badgeClass = 'px-2 py-1 rounded';
                
                let statsHtml = `
                    <tr><td class="font-medium text-muted w-1/3">Data Type</td><td>${stats.dtype}</td></tr>
                    <tr><td class="font-medium text-muted">Total Records</td><td>${stats.total_records}</td></tr>
                    <tr><td class="font-medium text-muted">Missing Count</td><td>${stats.missing_count}</td></tr>
                    <tr><td class="font-medium text-muted">Non-Missing Count</td><td>${stats.non_missing_count}</td></tr>
                    <tr><td class="font-medium text-muted">Missing Percentage</td><td>${stats.missing_pct}%</td></tr>
                    <tr><td class="font-medium text-muted">Completeness Percentage</td><td>${stats.completeness_pct}%</td></tr>
                    <tr><td class="font-medium text-muted align-middle">Status</td>
                        <td>
                            <span class="badge ${stats.status === 'Moderate Missingness' ? '' : badgeClass} px-2 py-1 rounded" 
                                  style="font-size: 0.75rem; ${stats.status === 'Moderate Missingness' ? 'background-color: #fef3c7; color: #b45309;' : ''}">
                                ${stats.status}
                            </span>
                        </td>
                    </tr>
                `;
                
                inspectStatsBody.innerHTML = statsHtml;
            })
            .catch(error => {
                inspectSpinner.classList.add('hidden');
                inspectError.textContent = 'An error occurred while fetching feature statistics.';
                inspectError.classList.remove('hidden');
                console.error(error);
            });
        });
    }
});
