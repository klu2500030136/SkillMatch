document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // Update button styles
            tabBtns.forEach(b => {
                b.classList.remove('border-primary', 'text-primary');
                b.classList.add('border-transparent', 'text-muted');
            });
            btn.classList.remove('border-transparent', 'text-muted');
            btn.classList.add('border-primary', 'text-primary');
            
            // Show/hide content
            tabContents.forEach(content => {
                content.classList.add('hidden');
            });
            document.getElementById(`tab-${tab}`).classList.remove('hidden');
        });
    });
    
    // Problem type switching (supervised)
    const problemTypeRadios = document.querySelectorAll('input[name="problem_type"]');
    const sectionClassification = document.getElementById('section-classification');
    const sectionRegression = document.getElementById('section-regression');
    
    if (problemTypeRadios.length > 0) {
        problemTypeRadios.forEach(radio => {
            radio.addEventListener('change', () => {
                if (radio.value === 'classification') {
                    sectionClassification.classList.remove('hidden');
                    sectionRegression.classList.add('hidden');
                } else {
                    sectionClassification.classList.add('hidden');
                    sectionRegression.classList.remove('hidden');
                }
            });
        });
    }
    
    // Unsupervised algorithm selection
    const unsupervisedBtns = document.querySelectorAll('.unsupervised-btn');
    const algoSections = document.querySelectorAll('.algo-section');
    
    unsupervisedBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const algo = btn.dataset.algo;
            
            // Update button styles
            unsupervisedBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show/hide sections
            algoSections.forEach(section => {
                section.classList.add('hidden');
            });
            document.getElementById(`algo-${algo}`).classList.remove('hidden');
        });
    });
    
    // Linear Regression Training
    const btnTrainRegression = document.getElementById('btn-train-regression');
    if (btnTrainRegression) {
        btnTrainRegression.addEventListener('click', () => {
            const randomState = document.getElementById('regression-random-state').value;
            
            btnTrainRegression.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Training...';
            btnTrainRegression.disabled = true;
            
            fetch('/ml/regression/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    target_col: 'YearsAtCompany',
                    random_state: randomState
                })
            })
            .then(res => res.json())
            .then(data => {
                btnTrainRegression.innerHTML = '<i class="ph ph-play"></i> Train Linear Regression';
                btnTrainRegression.disabled = false;
                
                if (data.error) {
                    alert('Regression Error: ' + data.error);
                    return;
                }
                
                // Show results
                document.getElementById('regression-results').classList.remove('hidden');
                document.getElementById('reg-mae').textContent = data.model.mae;
                document.getElementById('reg-mse').textContent = data.model.mse;
                document.getElementById('reg-rmse').textContent = data.model.rmse;
                document.getElementById('reg-r2').textContent = data.model.r2;
            })
            .catch(err => {
                console.error(err);
                btnTrainRegression.innerHTML = '<i class="ph ph-play"></i> Train Linear Regression';
                btnTrainRegression.disabled = false;
                alert('An unexpected error occurred.');
            });
        });
    }
    
    // K-Means
    const btnRunKmeans = document.getElementById('btn-run-kmeans');
    if (btnRunKmeans) {
        btnRunKmeans.addEventListener('click', () => {
            const k = parseInt(document.getElementById('kmeans-k').value);
            const usePca = document.getElementById('kmeans-representation').value === 'true';
            const randomState = parseInt(document.getElementById('kmeans-random-state').value);
            
            btnRunKmeans.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Running...';
            btnRunKmeans.disabled = true;
            
            fetch('/ml/unsupervised/kmeans', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    n_clusters: k,
                    use_pca: usePca,
                    random_state: randomState
                })
            })
            .then(res => res.json())
            .then(data => {
                btnRunKmeans.innerHTML = '<i class="ph ph-play"></i> Run K-Means';
                btnRunKmeans.disabled = false;
                
                if (data.error) {
                    alert('K-Means Error: ' + data.error);
                    return;
                }
                
                // Show results
                document.getElementById('kmeans-results').classList.remove('hidden');
                
                // Cluster distribution
                let distHtml = '<div class="text-sm">';
                for (const [cluster, info] of Object.entries(data.cluster_distribution)) {
                    distHtml += `<div class="flex justify-between mb-1">
                        <span>Cluster ${cluster}</span>
                        <span>${info.count} (${info.percentage}%)</span>
                    </div>`;
                }
                distHtml += '</div>';
                document.getElementById('kmeans-cluster-dist').innerHTML = distHtml;
                
                // Cluster characteristics
                let charHtml = '<div class="text-sm">';
                for (const [cluster, chars] of Object.entries(data.cluster_characteristics)) {
                    charHtml += `<div class="mb-2"><strong>Cluster ${cluster}:</strong><br>`;
                    for (const [feat, val] of Object.entries(chars)) {
                        charHtml += `${feat}: ${val}<br>`;
                    }
                    charHtml += '</div>';
                }
                charHtml += '</div>';
                document.getElementById('kmeans-cluster-char').innerHTML = charHtml;
                
                // Visualization
                if (data.visualization_data) {
                    drawKmeansChart(data.visualization_data);
                }
            })
            .catch(err => {
                console.error(err);
                btnRunKmeans.innerHTML = '<i class="ph ph-play"></i> Run K-Means';
                btnRunKmeans.disabled = false;
                alert('An unexpected error occurred.');
            });
        });
    }
    
    // DBSCAN
    const btnRunDbscan = document.getElementById('btn-run-dbscan');
    if (btnRunDbscan) {
        btnRunDbscan.addEventListener('click', () => {
            const eps = parseFloat(document.getElementById('dbscan-eps').value);
            const minSamples = parseInt(document.getElementById('dbscan-min-samples').value);
            const usePca = document.getElementById('dbscan-representation').value === 'true';
            
            btnRunDbscan.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Running...';
            btnRunDbscan.disabled = true;
            
            fetch('/ml/unsupervised/dbscan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    eps: eps,
                    min_samples: minSamples,
                    use_pca: usePca
                })
            })
            .then(res => res.json())
            .then(data => {
                btnRunDbscan.innerHTML = '<i class="ph ph-play"></i> Run DBSCAN';
                btnRunDbscan.disabled = false;
                
                if (data.error) {
                    alert('DBSCAN Error: ' + data.error);
                    return;
                }
                
                // Show results
                document.getElementById('dbscan-results').classList.remove('hidden');
                document.getElementById('dbscan-clusters').textContent = data.n_clusters;
                document.getElementById('dbscan-noise').textContent = data.n_noise;
                document.getElementById('dbscan-noise-pct').textContent = data.noise_percentage + '%';
                document.getElementById('dbscan-rep').textContent = data.representation;
                
                // Cluster distribution
                let distHtml = '<div class="text-sm">';
                for (const [cluster, info] of Object.entries(data.cluster_distribution)) {
                    const label = cluster === -1 ? 'Noise' : `Cluster ${cluster}`;
                    distHtml += `<div class="flex justify-between mb-1">
                        <span>${label}</span>
                        <span>${info.count} (${info.percentage}%)</span>
                    </div>`;
                }
                distHtml += '</div>';
                document.getElementById('dbscan-cluster-dist').innerHTML = distHtml;
                
                // Visualization
                if (data.visualization_data) {
                    drawDbscanChart(data.visualization_data);
                }
            })
            .catch(err => {
                console.error(err);
                btnRunDbscan.innerHTML = '<i class="ph ph-play"></i> Run DBSCAN';
                btnRunDbscan.disabled = false;
                alert('An unexpected error occurred.');
            });
        });
    }
    
    // Anomaly Detection
    const btnRunAnomaly = document.getElementById('btn-run-anomaly');
    if (btnRunAnomaly) {
        btnRunAnomaly.addEventListener('click', () => {
            const contamination = parseFloat(document.getElementById('anomaly-contamination').value);
            const usePca = document.getElementById('anomaly-representation').value === 'true';
            const randomState = parseInt(document.getElementById('anomaly-random-state').value);
            
            btnRunAnomaly.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Detecting...';
            btnRunAnomaly.disabled = true;
            
            fetch('/ml/unsupervised/anomaly', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    contamination: contamination,
                    use_pca: usePca,
                    random_state: randomState
                })
            })
            .then(res => res.json())
            .then(data => {
                btnRunAnomaly.innerHTML = '<i class="ph ph-play"></i> Detect Anomalies';
                btnRunAnomaly.disabled = false;
                
                if (data.error) {
                    alert('Anomaly Detection Error: ' + data.error);
                    return;
                }
                
                // Show results
                document.getElementById('anomaly-results').classList.remove('hidden');
                document.getElementById('anomaly-total').textContent = data.n_normal + data.n_anomalies;
                document.getElementById('anomaly-normal').textContent = data.n_normal;
                document.getElementById('anomaly-count').textContent = data.n_anomalies;
                document.getElementById('anomaly-pct').textContent = data.anomaly_percentage + '%';
                
                // Visualization
                if (data.visualization_data) {
                    drawAnomalyChart(data.visualization_data);
                }
            })
            .catch(err => {
                console.error(err);
                btnRunAnomaly.innerHTML = '<i class="ph ph-play"></i> Detect Anomalies';
                btnRunAnomaly.disabled = false;
                alert('An unexpected error occurred.');
            });
        });
    }
    
    // Chart instances
    let kmeansChartInstance = null;
    let dbscanChartInstance = null;
    let anomalyChartInstance = null;
    
    function drawKmeansChart(data) {
        const ctx = document.getElementById('kmeans-chart').getContext('2d');
        
        if (kmeansChartInstance) {
            kmeansChartInstance.destroy();
        }
        
        // Group by cluster
        const clusters = {};
        data.forEach(point => {
            if (!clusters[point.cluster]) {
                clusters[point.cluster] = [];
            }
            clusters[point.cluster].push({x: point.pc1, y: point.pc2});
        });
        
        const datasets = [];
        const colors = ['#3AAFA9', '#FF7043', '#FFCA28', '#AB47BC', '#EF5350', '#26A69A', '#78909C', '#8D6E63', '#5C6BC0', '#EC407A'];
        
        Object.keys(clusters).forEach((cluster, i) => {
            datasets.push({
                label: `Cluster ${cluster}`,
                data: clusters[cluster],
                backgroundColor: colors[i % colors.length],
                pointRadius: 4
            });
        });
        
        kmeansChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'PC1' }
                    },
                    y: {
                        title: { display: true, text: 'PC2' }
                    }
                }
            }
        });
    }
    
    function drawDbscanChart(data) {
        const ctx = document.getElementById('dbscan-chart').getContext('2d');
        
        if (dbscanChartInstance) {
            dbscanChartInstance.destroy();
        }
        
        // Separate noise from clusters
        const clusters = {};
        const noise = [];
        
        data.forEach(point => {
            if (point.cluster === -1) {
                noise.push({x: point.pc1, y: point.pc2});
            } else {
                if (!clusters[point.cluster]) {
                    clusters[point.cluster] = [];
                }
                clusters[point.cluster].push({x: point.pc1, y: point.pc2});
            }
        });
        
        const datasets = [];
        const colors = ['#3AAFA9', '#FF7043', '#FFCA28', '#AB47BC', '#EF5350', '#26A69A', '#78909C', '#8D6E63', '#5C6BC0', '#EC407A'];
        
        Object.keys(clusters).forEach((cluster, i) => {
            datasets.push({
                label: `Cluster ${cluster}`,
                data: clusters[cluster],
                backgroundColor: colors[i % colors.length],
                pointRadius: 4
            });
        });
        
        if (noise.length > 0) {
            datasets.push({
                label: 'Noise',
                data: noise,
                backgroundColor: '#9E9E9E',
                pointRadius: 4
            });
        }
        
        dbscanChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'PC1' }
                    },
                    y: {
                        title: { display: true, text: 'PC2' }
                    }
                }
            }
        });
    }
    
    function drawAnomalyChart(data) {
        const ctx = document.getElementById('anomaly-chart').getContext('2d');
        
        if (anomalyChartInstance) {
            anomalyChartInstance.destroy();
        }
        
        // Separate normal from anomalies
        const normal = [];
        const anomalies = [];
        
        data.forEach(point => {
            if (point.label === -1) {
                anomalies.push({x: point.pc1, y: point.pc2});
            } else {
                normal.push({x: point.pc1, y: point.pc2});
            }
        });
        
        const datasets = [
            {
                label: 'Normal',
                data: normal,
                backgroundColor: '#3AAFA9',
                pointRadius: 4
            },
            {
                label: 'Anomaly',
                data: anomalies,
                backgroundColor: '#FF7043',
                pointRadius: 6
            }
        ];
        
        anomalyChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'PC1' }
                    },
                    y: {
                        title: { display: true, text: 'PC2' }
                    }
                }
            }
        });
    }
});
