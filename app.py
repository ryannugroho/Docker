from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # Jika menggunakan model yang telah dilatih

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load data dan scaler
scaler = StandardScaler()
data_training = pd.read_csv('data_balita_bersih2.csv')
X_train = data_training[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]

scaler.fit(X_train[['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin']])

# Memuat model yang sudah dilatih (pastikan model sudah disimpan dalam format .h5 atau lainnya)
model = load_model('model_gizi.h5')

# Fungsi untuk prediksi
def predict_status_gizi(umur, tinggi_badan, jenis_kelamin):
    jenis_kelamin_map = {'laki-laki': 0, 'perempuan': 1}
    jenis_kelamin_input = jenis_kelamin_map[jenis_kelamin]

    input_data = np.array([[umur, tinggi_badan, jenis_kelamin_input]])
    input_data_scaled = scaler.transform(input_data)

    # Prediksi status gizi dengan model yang sudah dilatih
    y_pred = model.predict(input_data_scaled)
    y_pred_label = np.argmax(y_pred, axis=1)[0]

    status_gizi_map = {
        0: 'Normal',
        1: 'Severely Stunted',
        2: 'Stunted',
        3: 'Tinggi'
    }
    status_gizi = status_gizi_map[y_pred_label]

    return status_gizi

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        umur = int(request.form['umur'])
        tinggi_badan = float(request.form['tinggi_badan'])
        jenis_kelamin = request.form['jenis_kelamin']

        status_gizi_predicted = predict_status_gizi(umur, tinggi_badan, jenis_kelamin)
        return render_template('index.html', status_gizi=status_gizi_predicted)

    return render_template('index.html', status_gizi=None)

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)