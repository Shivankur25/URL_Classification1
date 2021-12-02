import numpy as np 
import pandas as pd
from flask import Flask,request,jsonify,render_template
import joblib
app=Flask(_name_)
model=joblib.load("nb_learner_model.pkl")
tf_vector=joblib.load("tf_vector_model.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = [request.form['review']]
    output=model.predict(tf_vector.transform(text))==0
    if output[0]==True:
        result='Good'
    else:
        result='Bad'
    return render_template('analysis.html', prediction_text='Predicted Result:   {}'.format(result))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(tf_vector.transform(list(data.values())))
    output = prediction[0]
    return jsonify(output)

if _name_ == "_main_":
    app.run(debug=True)