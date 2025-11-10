
import pickle


input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

def predict(customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred
