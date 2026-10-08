from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fruits.pkl", "rb"))

# Load label encoder
encoder = pickle.load(open("label_encoder.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    diameter = float(request.form["diameter"])
    weight = float(request.form["weight"])

    # Input for ML model
    input_data = np.array([[diameter, weight]])

    # Prediction
    prediction = model.predict(input_data)

    # Convert number into fruit name
    fruit_name = encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction=fruit_name
    )


if __name__ == "__main__":
    app.run(debug=True)
    