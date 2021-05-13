import os
from flask import Flask, request, abort, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from flask_ngrok import run_with_ngrok
import logging
#from gevent.pywsgi import WSGIServer
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
arrs = []
MODEL_PATH = "./model_files/Epiassist_model3.h5"
Threshold =   1.21249 # Threshold set after training the model (set at 3 Std deviations from the mean reconstruction error)
m = tf.keras.models.load_model(MODEL_PATH)

def flatten(tensor):
    flattened_X = np.empty((tensor.shape[0], tensor.shape[2]))  # sample x features array.
    for i in range(tensor.shape[0]):
        flattened_X[i] = tensor[i, (tensor.shape[1] - 1), :]
    return (flattened_X)

application = Flask(__name__)
run_with_ngrok(application)
@application.route("/",methods=["GET"])
def start():
    #try:
	  return ("Epiassist_API is working" , 200)
    #except:
     # return (404)

@application.route("/gdap",methods=["POST"])

def gdap():
    #arrs =[]
    oxy = float(request.args.get('oxy'))
    temp= float(request.args.get('temp'))
    #emg = float(request.args.get('emg'))
    hrt = float(request.args.get('hrt'))
    acc = float(request.args.get('acc'))
    arr = [oxy , temp , hrt , acc]
    #pred = m.predict(arr)
    arrs.append(arr)
    print(application.logger.info(arr))
    #return jsonify(arrs)
    if len(arrs)>= 10:
          X =np.array(arrs[len(arrs)-10: len(arrs)])
          X = StandardScaler().fit_transform(X)
          X =np.reshape(X ,(1,10,4))
          pred= m.predict(X)
          err = tf.keras.losses.mae(flatten(X), flatten(pred))
          #print(np.shape(X))
          if err>=Threshold:
              return ("Seizure is near!")
              #print(err)
          else:
             return("Normal")

    return "Get more data"

if __name__ == '__main__':
    application.run()






