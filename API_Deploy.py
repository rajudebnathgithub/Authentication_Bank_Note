from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("BankNote.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Bank Note Authentication
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Variance
        in: query
        type: number
        required: true
      - name: Skewness
        in: query
        type: number
        required: true
      - name: Curtosis
        in: query
        type: number
        required: true
      - name: Entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Prediction is
        
    """
    Variance=request.args.get("Variance")
    Skewness=request.args.get("Skewness")
    Curtosis=request.args.get("Curtosis")
    Entropy=request.args.get("Entropy")
    prediction=classifier.predict([[Variance,Skewness,Curtosis,Entropy]])
    print(prediction)
    return "Prediction is "+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Bank Note Authentication
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The Prediction is
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)