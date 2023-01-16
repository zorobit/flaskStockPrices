from flask import Flask, request, jsonify, request,render_template,jsonify
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
app = Flask(__name__)

# Load the trained LSTM model
model = load_model("models/stock_price_model.h5")
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the request
    ticker = request.form.get("ticker")
    start_date = request.form.get("start-date")
    end_date = request.form.get("end-date")

    # Download historical stock prices for a specific ticker
    df = yf.download(ticker, start=start_date, end=end_date)
    new_df = df.filter(['Close'])
    last_60_days = new_df[-60:].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_60_days_scaled=scaler.fit_transform(last_60_days)
    data= []
    data.append(last_60_days_scaled)
    data = np.array(data)
    data = np.reshape(data,(data.shape[0],1,data.shape[1]))


    print(data.shape)

    predictions = model.predict(data)

    predictions = scaler.inverse_transform(predictions)

    return jsonify({'pred':str(predictions)})


if __name__ == "__main__":
    app.run(debug=True)