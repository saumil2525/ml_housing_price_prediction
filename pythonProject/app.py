import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from joblib.parallel import method

# initialize flask app
app = Flask(__name__)

# import model
model = pickle.load(open("models/regmodel.pkl", "rb"))
scalar = pickle.load(open("models/scaling.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data_dict = request.json["data"]
    my_list = [
        "OverallQual",
        "YearBuilt",
        "YearRemodAdd",
        "TotalBsmtSF",
        "1stFlrSF",
        "GrLivArea",
        "FullBath",
        "TotRmsAbvGrd",
        "GarageCars",
        "GarageArea",
        "ExterQual_TA",
        "BsmtQual_Ex",
        "KitchenQual_Ex",
        "KitchenQual_TA",
        "GarageFinish_Unf",
    ]
    data = {k: v for k, v in data_dict.items() if k in my_list}
    print("data: ", data)
    print(np.array(list(data.values())).reshape(1, -1))  # single row
    new_data = scalar.transform(
        np.array(list(data.values())).reshape(1, -1)
    )  # scale data
    output = model.predict(new_data)
    print("prediction: ", output[0])
    return jsonify({"prediction": output[0]})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text="Predicted House Price: {output}")


if __name__ == "__main__":
    app.run(debug=True)
