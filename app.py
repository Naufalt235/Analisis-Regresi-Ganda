from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
import traceback

app = Flask(__name__)

# ============================================
# FUNGSI UNTUK MEMBUAT GAMBAR BASE64
# ============================================
def plot_to_base64():
    img = BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

# ============================================
# ROUTE UTAMA
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

# ============================================
# API UNTUK ANALISIS DATA
# ============================================
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        print("📊 Memulai analisis data...")
        
        # Baca data
        df = pd.read_csv('HARGA RUMAH JAKSEL.csv')
        print(f"✅ Data terbaca: {len(df)} records")
        
        # Bersihkan data
        df.columns = df.columns.str.strip()
        df['GRS'] = df['GRS'].map({'ADA': 1, 'TIDAK ADA': 0})
        df['HARGA'] = pd.to_numeric(df['HARGA'], errors='coerce')
        df['LT'] = pd.to_numeric(df['LT'], errors='coerce')
        df['LB'] = pd.to_numeric(df['LB'], errors='coerce')
        df['JKT'] = pd.to_numeric(df['JKT'], errors='coerce')
        df['JKM'] = pd.to_numeric(df['JKM'], errors='coerce')
        df = df.dropna()
        print(f"✅ Data setelah cleaning: {len(df)} records")
        
        # Ambil SEMUA data untuk tabel (TIDAK DIBATASI 100)
        data_table = df.to_dict(orient='records')
        
        # ============================================
        # MODEL REGRESI (4 VARIABEL: LT, LB, JKT, GRS)
        # ============================================
        X = df[['LT', 'LB', 'JKT', 'GRS']]
        y = df['HARGA']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # ============================================
        # METRIK EVALUASI
        # ============================================
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # ============================================
        # KONTRIBUSI RELATIF (SCALED COEFFICIENT)
        # ============================================
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_scaled = LinearRegression()
        model_scaled.fit(X_scaled, y)
        
        coef_abs = np.abs(model_scaled.coef_)
        total_abs = np.sum(coef_abs)
        kontribusi = (coef_abs / total_abs * 100).tolist()
        
        koefisien = {
            'intercept': model.intercept_,
            'LT': model.coef_[0],
            'LB': model.coef_[1],
            'JKT': model.coef_[2],
            'GRS': model.coef_[3],
            'kontribusi': kontribusi
        }
        
        print("📊 Membuat gambar...")
        gambar = {}
        
        # 1. Heatmap (semua variabel)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[['HARGA', 'LT', 'LB', 'JKT', 'JKM', 'GRS']].corr(), 
                    annot=True, cmap='coolwarm', fmt='.2f', 
                    square=True, linewidths=0.5)
        plt.title('Heatmap Korelasi - Harga Rumah Jaksel')
        plt.tight_layout()
        gambar['heatmap'] = plot_to_base64()
        
        # 2. Scatter LT
        plt.figure(figsize=(6, 4))
        plt.scatter(df['LT'], df['HARGA'], alpha=0.5, s=10)
        plt.xlabel('Luas Tanah (m²)')
        plt.ylabel('Harga (Rp)')
        plt.title('LT vs HARGA')
        plt.tight_layout()
        gambar['scatter_lt'] = plot_to_base64()
        
        # 3. Scatter LB
        plt.figure(figsize=(6, 4))
        plt.scatter(df['LB'], df['HARGA'], alpha=0.5, s=10)
        plt.xlabel('Luas Bangunan (m²)')
        plt.ylabel('Harga (Rp)')
        plt.title('LB vs HARGA')
        plt.tight_layout()
        gambar['scatter_lb'] = plot_to_base64()
        
        # 4. Scatter JKT
        plt.figure(figsize=(6, 4))
        plt.scatter(df['JKT'], df['HARGA'], alpha=0.5, s=10)
        plt.xlabel('Kamar Tidur')
        plt.ylabel('Harga (Rp)')
        plt.title('JKT vs HARGA')
        plt.tight_layout()
        gambar['scatter_jkt'] = plot_to_base64()
        
        # 5. Scatter GRS
        plt.figure(figsize=(6, 4))
        plt.scatter(df['GRS'], df['HARGA'], alpha=0.5, s=10)
        plt.xlabel('Garasi (0=Tidak, 1=Ada)')
        plt.ylabel('Harga (Rp)')
        plt.title('GRS vs HARGA')
        plt.tight_layout()
        gambar['scatter_grs'] = plot_to_base64()
        
        # 6. Actual vs Predicted
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.5, s=10)
        plt.plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Aktual')
        plt.ylabel('Prediksi')
        plt.title(f'Actual vs Predicted (R² = {r2:.4f})')
        plt.tight_layout()
        gambar['actual_pred'] = plot_to_base64()
        
        # 7. Residual Plot
        residual = y_test - y_pred
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residual, alpha=0.5, s=10)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Prediksi')
        plt.ylabel('Residual')
        plt.title('Residual Plot')
        plt.tight_layout()
        gambar['residual'] = plot_to_base64()
        
        # 8. PLS Scores
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=20)
        plt.colorbar(scatter, label='Harga')
        plt.xlabel('Komponen 1')
        plt.ylabel('Komponen 2')
        plt.title('PLS Scores Plot (PCA)')
        plt.tight_layout()
        gambar['pls_scores'] = plot_to_base64()
        
        # 9. PLS Loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        plt.figure(figsize=(6, 5))
        for i, var in enumerate(['LT', 'LB', 'JKT', 'GRS']):
            plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                     head_width=0.05, head_length=0.05, fc='red', ec='red')
            plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, var, fontsize=12)
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Komponen 1')
        plt.ylabel('Komponen 2')
        plt.title('PLS Loadings')
        plt.tight_layout()
        gambar['pls_loadings'] = plot_to_base64()
        
        print("✅ Semua gambar selesai!")
        
        return jsonify({
            'success': True,
            'data_table': data_table,
            'total_data': len(df),
            'metrics': {
                'r2': r2,
                'adj_r2': adj_r2,
                'rmse': rmse,
                'mae': mae
            },
            'koefisien': koefisien,
            'gambar': gambar
        })
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ============================================
# API UNTUK PREDIKSI
# ============================================
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        LT = float(data.get('LT', 0))
        LB = float(data.get('LB', 0))
        JKT = float(data.get('JKT', 0))
        GRS = float(data.get('GRS', 0))
        
        df = pd.read_csv('HARGA RUMAH JAKSEL.csv')
        df.columns = df.columns.str.strip()
        df['GRS'] = df['GRS'].map({'ADA': 1, 'TIDAK ADA': 0})
        df['HARGA'] = pd.to_numeric(df['HARGA'], errors='coerce')
        df['LT'] = pd.to_numeric(df['LT'], errors='coerce')
        df['LB'] = pd.to_numeric(df['LB'], errors='coerce')
        df['JKT'] = pd.to_numeric(df['JKT'], errors='coerce')
        df = df.dropna()
        
        X = df[['LT', 'LB', 'JKT', 'GRS']]
        y = df['HARGA']
        
        model = LinearRegression()
        model.fit(X, y)
        
        prediksi = model.predict([[LT, LB, JKT, GRS]])[0]
        
        return jsonify({
            'success': True,
            'prediksi': float(prediksi)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("="*50)
    print("MENJALANKAN APLIKASI WEB...")
    print("Data: HARGA RUMAH JAKSEL.csv")
    print("Variabel X: LT, LB, JKT, GRS")
    print("Buka: http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True)