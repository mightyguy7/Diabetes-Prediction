from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open(b'C:\Users\Aditya Rawat\diabetes_model.pkl','rb'))
scaler = pickle.load(open(b'C:\Users\Aditya Rawat\Downloads\scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('myindex.html')  # No need for absolute paths

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[key]) for key in request.form.keys()]
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    return render_template('result.html', result=result)  # No need for absolute paths

if __name__ == '__main__':
    app.run()
