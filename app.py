from flask import Flask, render_template, request
from flask_cors import CORS
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

app = Flask(__name__)

CORS(app)

# Load the trained model
model = tf.keras.models.load_model('model_status_gizi.h5')

# Initialize StandardScaler
scaler = StandardScaler()

# Load training data for scaling
data_training = pd.read_csv('data_balita_bersih2.csv')
X_train = data_training[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]

# Preprocess 'Jenis Kelamin' for scaling (same as in your original code)
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].str.strip().str.lower()
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})
data_training['Jenis Kelamin'] = data_training['Jenis Kelamin'].fillna(-1)
X_train = data_training[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]

# Fit the scaler
scaler.fit(X_train) 

# Prediction function
def predict_status_gizi(umur, tinggi_badan, jenis_kelamin):
    jenis_kelamin_map = {'laki-laki': 0, 'perempuan': 1}
    jenis_kelamin_input = jenis_kelamin_map[jenis_kelamin]
    input_data = np.array([[umur, tinggi_badan, jenis_kelamin_input]])
    input_data_scaled = scaler.transform(input_data)
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

@app.route('/predict', methods=['POST'])
def predict():
    umur = int(request.form['umur'])
    tinggi_badan = float(request.form['tinggi_badan'])
    jenis_kelamin = request.form['jenis_kelamin']
    prediction = predict_status_gizi(umur, tinggi_badan, jenis_kelamin)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
