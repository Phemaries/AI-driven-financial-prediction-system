from flask import Flask, jsonify, request
import pickle
import pandas as pd

import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "..", "predictions", "dataset", "stocks.pkl")
model_path = os.path.abspath(model_path)

if not os.path.exists(model_path):
    print(f"[WARN] Model file not found at {model_path}")
    model, model_xgb = None, None
else:
    with open(model_path, "rb") as f:
        model, model_xgb = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model_xgb.predict(df)
    return jsonify({"Predicted 5-day Volatility Estimate": f"{round((float(prediction) * 100), 2)}%"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
