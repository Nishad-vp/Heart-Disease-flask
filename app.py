from flask import Flask,render_template,request
import numpy as np
import joblib
from joblib import load
app=Flask(__name__)

model = load(filename='Heart-Disease-model.joblib')
# Route

@app.route("/")
def home():
    return render_template('index.html')



@app.route("/status", methods=['POST'])
def status():
    if request.method=="POST":
        age = int(request.form['age'])
        sex= int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach  = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak =int( request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal =int(request.form['thal'])

        input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        ## Change to numpy array
        input_data_array= np.asarray(input_data)

        #Reshape
        input_data_reshape = input_data_array.reshape(1,-1)
        prediction=model.predict(input_data_reshape)
        if prediction[0]==0:
            result = "No Heart Disease"
        else:
            result="Heart Disease"
    return render_template('result.html',result=result)    

    

if __name__== '__main__':
    app.run(debug=True)