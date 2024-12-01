from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load model dan scaler
model = load_model('model_status_gizi.keras')
scaler = StandardScaler()

# Load dan persiapkan data untuk fitting scaler
data= pd.read_csv('https://raw.githubusercontent.com/ryannugroho/Docker/refs/heads/main/status_gizi_clean2.csv')
label_encoder = LabelEncoder()
data['JK'] = label_encoder.fit_transform(data['JK'])

# Fit scaler dengan data pelatihan
X_train = data[['Usia', 'Berat', 'Tinggi', 'LiLA', 'JK']]
scaler.fit(X_train[['Usia', 'Berat', 'Tinggi', 'LiLA', 'JK']])

# Fungsi prediksi status gizi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Ambil data input dari POST request
        
        # Ambil input data
        umur = int(data['umur'])
        tinggi_badan = float(data['tinggi_badan'])
        berat_badan = float(data['berat_badan'])
        lila = float(data['lila'])
        jenis_kelamin = data['jenis_kelamin'].lower()

        # Map gender to numeric
        jenis_kelamin_map = {'laki-laki': 0, 'perempuan': 1}
        if jenis_kelamin not in jenis_kelamin_map:
            return jsonify({'error': 'Jenis kelamin harus "laki-laki" atau "perempuan".'}), 400

        jenis_kelamin_input = jenis_kelamin_map[jenis_kelamin]

        # Input data untuk prediksi
        input_data = np.array([[umur, tinggi_badan, berat_badan, lila, jenis_kelamin_input]])

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
            0: 'Gizi Baik',
            1: 'Gizi Buruk',
            2: 'Gizi Kurang',
            3: 'Gizi Lebih',
            4: 'Obesitas',
            5: 'Resiko Gizi Lebih'
        }
        status_gizi = status_gizi_map.get(y_pred_label, 'Unknown')

        # Kembalikan hasil prediksi
        return jsonify({'status_gizi': status_gizi})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
