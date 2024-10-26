import pickle as pkl

with open("model1.bin", "rb") as f:
    model = pkl.load(f)

with open("dv.bin", "rb") as f:
    dv = pkl.load(f)


input = dv.transform([{"job": "management", "duration": 400, "poutcome": "success"}])

print(model.predict_proba(input))