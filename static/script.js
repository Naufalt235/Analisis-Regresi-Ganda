document.addEventListener('DOMContentLoaded', function() {

    // Data asli untuk filtering
    let originalData = [];
    let filteredData = [];

    function loadAnalysis() {
        fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                originalData = data.data_table;
                filteredData = [...originalData];
                
                document.getElementById('total-data').textContent = data.total_data + ' records';
                document.getElementById('total-data-table').textContent = data.total_data;
                
                document.getElementById('r2-value').textContent = data.metrics.r2.toFixed(4);
                document.getElementById('adj-r2-value').textContent = data.metrics.adj_r2.toFixed(4);
                document.getElementById('rmse-value').textContent = 'Rp ' + formatNumber(data.metrics.rmse);
                document.getElementById('mae-value').textContent = 'Rp ' + formatNumber(data.metrics.mae);
                
                const k = data.koefisien;
                document.getElementById('model-result').textContent = 
                    `Harga = ${formatNumber(k.intercept)} + (${formatNumber(k.LT)} × LT) + (${formatNumber(k.LB)} × LB) + (${formatNumber(k.JKT)} × JKT) + (${formatNumber(k.GRS)} × GRS)`;
                
                document.getElementById('coef-lt-value').textContent = formatNumber(k.LT);
                document.getElementById('coef-lb-value').textContent = formatNumber(k.LB);
                document.getElementById('coef-jkt-value').textContent = formatNumber(k.JKT);
                document.getElementById('coef-grs-value').textContent = formatNumber(k.GRS);
                
                document.getElementById('bar-lt').style.width = k.kontribusi[0] + '%';
                document.getElementById('bar-lb').style.width = k.kontribusi[1] + '%';
                document.getElementById('bar-jkt').style.width = k.kontribusi[2] + '%';
                document.getElementById('bar-grs').style.width = k.kontribusi[3] + '%';
                
                document.getElementById('pct-lt').textContent = k.kontribusi[0].toFixed(1) + '%';
                document.getElementById('pct-lb').textContent = k.kontribusi[1].toFixed(1) + '%';
                document.getElementById('pct-jkt').textContent = k.kontribusi[2].toFixed(1) + '%';
                document.getElementById('pct-grs').textContent = k.kontribusi[3].toFixed(1) + '%';
                
                document.getElementById('heatmap-img').src = 'data:image/png;base64,' + data.gambar.heatmap;
                document.getElementById('scatter-lt-img').src = 'data:image/png;base64,' + data.gambar.scatter_lt;
                document.getElementById('scatter-lb-img').src = 'data:image/png;base64,' + data.gambar.scatter_lb;
                document.getElementById('scatter-jkt-img').src = 'data:image/png;base64,' + data.gambar.scatter_jkt;
                document.getElementById('scatter-grs-img').src = 'data:image/png;base64,' + data.gambar.scatter_grs;
                document.getElementById('actual-pred-img').src = 'data:image/png;base64,' + data.gambar.actual_pred;
                document.getElementById('residual-img').src = 'data:image/png;base64,' + data.gambar.residual;
                document.getElementById('pls-scores-img').src = 'data:image/png;base64,' + data.gambar.pls_scores;
                document.getElementById('pls-loadings-img').src = 'data:image/png;base64,' + data.gambar.pls_loadings;
                
                applyFilters();
            } else {
                alert('Gagal memuat data: ' + data.error);
            }
        })
        .catch(error => {
            alert('Terjadi kesalahan: ' + error);
        });
    }
    
    function applyFilters() {
        const garasiFilter = document.getElementById('filter-garasi').value;
        const sortHarga = document.getElementById('sort-harga').value;
        const showRows = document.getElementById('show-rows').value;
        
        // Filter Garasi
        let result = [...originalData];
        if (garasiFilter !== 'semua') {
            result = result.filter(row => row.GRS == garasiFilter);
        }
        
        // Sort Harga
        if (sortHarga === 'asc') {
            result.sort((a, b) => a.HARGA - b.HARGA);
        } else {
            result.sort((a, b) => b.HARGA - a.HARGA);
        }
        
        // Simpan semua hasil filter (tanpa batasan baris)
        filteredData = result;
        document.getElementById('total-data-table').textContent = filteredData.length;
        
        // Tentukan jumlah baris yang ditampilkan
        let displayCount = filteredData.length;
        if (showRows !== 'semua') {
            const limit = parseInt(showRows);
            if (!isNaN(limit) && limit > 0) {
                displayCount = Math.min(limit, filteredData.length);
            }
        }
        document.getElementById('display-count').textContent = displayCount;
        
        // Ambil data sesuai limit
        const displayData = filteredData.slice(0, displayCount);
        buildTable(displayData);
    }
    
    function buildTable(data) {
        if (!data || data.length === 0) {
            document.getElementById('table-body').innerHTML = '<tr><td colspan="7" style="text-align:center;padding:20px;color:#8a9aaa;">Tidak ada data</td></tr>';
            return;
        }
        
        const headerRow = document.getElementById('table-header');
        const columns = Object.keys(data[0]);
        headerRow.innerHTML = columns.map(col => `<th>${col}</th>`).join('');
        
        const tbody = document.getElementById('table-body');
        tbody.innerHTML = data.map(row => {
            return `<tr>${columns.map(col => {
                let value = row[col];
                if (typeof value === 'number' && col === 'HARGA') {
                    value = 'Rp ' + formatNumber(value);
                } else if (typeof value === 'number') {
                    value = formatNumber(value);
                }
                return `<td>${value}</td>`;
            }).join('')}</tr>`;
        }).join('');
    }
    
    // Event Listener untuk Filter - otomatis saat dropdown berubah
    document.getElementById('filter-garasi').addEventListener('change', applyFilters);
    document.getElementById('sort-harga').addEventListener('change', applyFilters);
    document.getElementById('show-rows').addEventListener('change', applyFilters);
    
    // Tombol Terapkan (tetap ada untuk kenyamanan)
    document.getElementById('btn-apply-filter').addEventListener('click', applyFilters);
    
    // Prediksi
    document.getElementById('btn-prediksi').addEventListener('click', function() {
        const LT = parseFloat(document.getElementById('input-lt').value);
        const LB = parseFloat(document.getElementById('input-lb').value);
        const JKT = parseFloat(document.getElementById('input-jkt').value);
        const GRS = parseFloat(document.getElementById('input-grs').value);
        
        if (isNaN(LT) || isNaN(LB) || isNaN(JKT) || isNaN(GRS)) {
            alert('Masukkan nilai yang valid!');
            return;
        }
        
        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ LT, LB, JKT, GRS })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('prediksi-nilai').textContent = 'Rp ' + formatNumber(data.prediksi);
            } else {
                alert('Gagal memprediksi: ' + data.error);
            }
        })
        .catch(error => {
            alert('Terjadi kesalahan: ' + error);
        });
    });
    
    function formatNumber(num) {
        if (num === undefined || num === null) return '0';
        const rounded = Math.round(num);
        return rounded.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
    
    loadAnalysis();
});