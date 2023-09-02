from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            second=float(request.form.get('second')),
            third=float(request.form.get('third')),
            fourth=float(request.form.get('fourth')),
            fifth=float(request.form.get('fifth')),
            gender=request.form.get('gender')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predictor(pred_df)

        return render_template('home.html', results=result[0])
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)