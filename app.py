from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # Mengimpor fungsi untuk memuat model Keras
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Memuat model yang sudah dilatih sebelumnya
model = load_model('model_status_gizi.h5')

scaler = StandardScaler()

data_training = pd.read_csv('data_balita_bersih2.csv')
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].str.strip().str.lower()
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].fillna(-1)

X_train = data_training[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
scaler.fit(X_train)

# Fungsi untuk prediksi
def predict_status_gizi(umur, tinggi_badan, jenis_kelamin):
    jenis_kelamin_map = {'laki-laki': 0, 'perempuan': 1}
    jenis_kelamin_input = jenis_kelamin_map.get(jenis_kelamin.lower(), -1)  # Menangani kasus yang tidak valid

    input_data = np.array([[umur, tinggi_badan, jenis_kelamin_input]])
    input_data_scaled = scaler.transform(input_data)

    # Melakukan prediksi
    y_pred = model.predict(input_data_scaled)
    y_pred_label = np.argmax(y_pred, axis=1)[0]  # Ambil label prediksi

    status_gizi_map = {
        0: 'Normal',
        1: 'Severely Stunted',
        2: 'Stunted',
        3: 'Tinggi'
    }
    status_gizi = status_gizi_map.get(y_pred_label, 'Unknown')  # Menangani label yang tidak dikenal

    return status_gizi

# API Endpoint untuk menerima input dan mengembalikan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Mengambil data yang dikirim oleh frontend
    umur = data['umur']
    tinggi_badan = data['tinggi_badan']
    jenis_kelamin = data['jenis_kelamin']

    # Lakukan prediksi dan kembalikan hasilnya
    status_gizi_predicted = predict_status_gizi(umur, tinggi_badan, jenis_kelamin)
    return jsonify({'status_gizi': status_gizi_predicted})

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
