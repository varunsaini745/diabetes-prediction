from flask import Flask, render_template, request
import pickle
import numpy as np

# load random classifier model
filename = 'diabetes_prediction_RFC_model.pkl'
RFC = pickle.load(open(filename, 'rb'))

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        blood_pressure = int(request.form['bloodpressure'])
        skin_thikness= int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, blood_pressure, skin_thikness, insulin, bmi, dpf, age ]])
        y_prediction = RFC.predict(data)
        
        return render_template('result.html', prediction = y_prediction)
    
if __name__ == '__main__':
    app.run()
