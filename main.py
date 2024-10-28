from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model (replace 'model.pkl' with your model file)
with open('StudentMarkPrediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #Get input from the form
    Study_Hours = request.form['Study Hours']
    
    #Make a prediction using loaded model
    input_data = [[float(Study_Hours)]]
    reshaped_data = np.array(input_data).reshape(1,-1)
    
    prediction = model.predict(reshaped_data)

    #Pass the prediction value to template
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
