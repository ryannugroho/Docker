from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model dan scaler
model = load_model('model_status_gizi.h5')  # Pastikan file model.h5 ada di direktori yang sama
scaler = StandardScaler()

# Load dan persiapkan data untuk fitting scaler
data_training = pd.read_csv('https://raw.githubusercontent.com/ryannugroho/Docker/refs/heads/main/data_balita_bersih2.csv')
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].str.strip().str.lower()
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].fillna(-1)

# Fit scaler dengan data pelatihan
X_train = data_training[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
scaler.fit(X_train[['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin']])

# Fungsi prediksi status gizi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Ambil data input dari POST request
        
        # Ambil input data
        umur = int(data['umur'])
        tinggi_badan = float(data['tinggi_badan'])
        jenis_kelamin = data['jenis_kelamin'].lower()

        # Map gender to numeric
        jenis_kelamin_map = {'laki-laki': 0, 'perempuan': 1}
        if jenis_kelamin not in jenis_kelamin_map:
            return jsonify({'error': 'Jenis kelamin harus "laki-laki" atau "perempuan".'}), 400

        jenis_kelamin_input = jenis_kelamin_map[jenis_kelamin]

        # Input data untuk prediksi
        input_data = np.array([[umur, tinggi_badan, jenis_kelamin_input]])

        # Scaling input data
        input_data_scaled = scaler.transform(input_data)
        print(f"Input data after scaling: {input_data_scaled}")

        # Prediksi dengan model
        y_pred = model.predict(input_data_scaled)
        print(f"Prediction output: {y_pred}")

        # Ambil label prediksi
        y_pred_label = np.argmax(y_pred, axis=1)[0]
        print(f"Predicted label: {y_pred_label}")

        # Pemetaan label ke status gizi
        status_gizi_map = {
            0: 'Normal',
            1: 'Severely Stunted',
            2: 'Stunted',
            3: 'Tinggi'
        }
        status_gizi = status_gizi_map.get(y_pred_label, 'Unknown')

        # Kembalikan hasil prediksi
        return jsonify({'status_gizi': status_gizi})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
