from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict-price', methods=['POST'])
def predict_price():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    try:
        prediction = model.predict(df_scaled)[0]
        return jsonify({'price': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
