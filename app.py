from flask import Flask,request,redirect
from flask.templating import render_template
import numpy as np
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict' , methods =['GET','POST'])
def predict():

    if request.method =='POST':
         data = [float(data) for data in request.form.values() ]
         data = [np.array(data)]

         model = pickle.load(open('model.pkl','rb'))
         prediction = float(model.predict(data))
         prediction= str(round(prediction, 3))

         return render_template('predict.html',prediction=prediction)


    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)