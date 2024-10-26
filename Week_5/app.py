from flask import Flask
from flask import request
from flask import jsonify
import pickle as pkl

with open("model1.bin", "rb") as f:
    model = pkl.load(f)

with open("dv.bin", "rb") as f:
    dv = pkl.load(f)

app = Flask('customer')

def predict_single(customer, dv, model):
    X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    prediction = predict_single(customer, dv, model)
    subscription = prediction >= 0.5

    result = {
        'subscription_probability': round(float(prediction), 3), ## we need to cast numpy float type to python native float type
        'subscription': bool(subscription),  ## same as the line above, casting the value using bool method
    }

    return jsonify(result)  ## send back the data in json format to the user

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)