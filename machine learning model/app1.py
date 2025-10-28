from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(baby_data):
    gestation = float(["gestation"])
    parity = int(["parity"])
    age = float(["age"]) 
    height = float(["height"])
    weight = float(["weight"])
    smoke = float(["smoke"])

    cleaned_data = {
        'gestation': [gestation],
        'parity': [parity], 
        'age': [age],
        'height': [height],
        'weight': [weight],
        'smoke': [smoke]
    }
    return get_cleaned_data

@app.route("/", methods = ["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def get_prediction():
    baby_data_form = request.form
    baby_data_cleaned = get_cleaned_data(baby_data_form)

    baby_df = pd.DataFrame(baby_data_cleaned)
    
    with open('model/model.pkl', 'rb') as obj:
        model = pickle.load(obj)

    prediction = model.predict(baby_df)
    prediction = round(float(prediction), 2)

    response = {"Prediction": prediction}
    return render_template("index.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug = True)