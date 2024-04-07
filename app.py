from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Clump_thickness =float(request.form['Clump thickness'])
            Uniformity_of_cell_size =float(request.form['Uniformity of cell size'])
            cUniformity_of_cell_shape =float(request.form['Uniformity of cell shape'])
            Marginalnadhesion =float(request.form['Marginaln adhesion'])
            Single_epithelial_cell_size =float(request.form['Single epithelial cell size'])
            Bare_nuclei=float(request.form['Bare nuclei'])
            Bland_chromatin =float(request.form['Bland chromatin'])
            Normal_nucleoli =float(request.form['Normal nucleoli'])
            Mitoses =float(request.form['Mitoses'])
            
       
         
            data = [Clump_thickness,Uniformity_of_cell_size,cUniformity_of_cell_shape,Marginalnadhesion,Single_epithelial_cell_size,Bare_nuclei,Bland_chromatin,Normal_nucleoli,Mitoses]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')



if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)