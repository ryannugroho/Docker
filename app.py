from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Load model dan scaler (ganti dengan file model dan scaler Anda)
model = load_model('model_status_gizi.keras')

scaler = StandardScaler()
data= pd.read_csv('https://raw.githubusercontent.com/ryannugroho/Docker/refs/heads/main/status_gizi_clean2.csv')
label_encoder = LabelEncoder()
data['JK'] = label_encoder.fit_transform(data['JK'])

x = data[['Usia', 'Berat', 'Tinggi', 'LiLA', 'JK']]

scaler.fit(x)

# Peta untuk status gizi
status_gizi_map = {
    0: 'Gizi Baik',
    1: 'Gizi Buruk',
    2: 'Gizi Kurang',
    3: 'Gizi Lebih',
    4: 'Obesitas',
    5: 'Resiko Gizi Lebih'
}

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request JSON
        data = request.json
        usia = data['usia']
        tinggi = data['tinggi']
        berat = data['berat']
        lila = data['lila']
        jk = data['jk']

        # Konversi jenis kelamin
        jenis_kelamin_map = {'laki-laki': 0, 'perempuan': 1}
        jenis_kelamin_input = jenis_kelamin_map[jk]

        # Buat DataFrame untuk prediksi
        new_data = pd.DataFrame({
            'Usia': [usia],
            'Berat': [berat],
            'Tinggi': [tinggi],
            'LiLA': [lila],
            'JK': [jenis_kelamin_input]
        })

        # Scale data menggunakan scaler yang sama dengan saat training
        new_data_scaled = scaler.transform(new_data)

        # Lakukan prediksi
        predictions = model.predict(new_data_scaled)
        predicted_class = np.argmax(predictions)
        predicted_status_gizi = status_gizi_map[predicted_class]

        # Return hasil prediksi
        return jsonify({
            'predicted_status_gizi': predicted_status_gizi
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
