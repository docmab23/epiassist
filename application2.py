import os
from flask import Flask, request, abort, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from flask_ngrok import run_with_ngrok
import logging
# from gevent.pywsgi import WSGIServer
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import requests

url = "https://www.fast2sms.com/dev/bulkV2"

querystring = {"authorization": "ICORfgkyj9WBJ6ENMT4zobwiPQGXZLHla251An738mecSsrvhKFwMXC8D7zP2grjUT1xo0q6sJWdayE9",
               "message": "Seizure is near , please help the patient", "language": "english", "route": "q",
               "numbers": "9888953231,9783554288"}

headers = {
    'cache-control': "no-cache"
}

MODEL_PATH = "./model_files/Epi_5.h5"
Threshold = 4500  # Threshold set after training the model (set at 3 Std deviations from the mean reconstruction error)
m = tf.keras.models.load_model(MODEL_PATH)


def flatten(tensor):
    flattened_X = np.empty((tensor.shape[0], tensor.shape[2]))  # sample x features array.
    for i in range(tensor.shape[0]):
        flattened_X[i] = tensor[i, (tensor.shape[1] - 1), :]
    return (flattened_X)


application2 = Flask(__name__)
run_with_ngrok(application2)


@application2.route("/", methods=["GET", "POST"])
def start():
    # try:
    return ("Epiassist_API is working", 200)


# except:
# return (404)
arrs = []


@application2.route("/gdap", methods=["GET","POST"])
def gdap():
    data = request.args.get('data')

    arr =[float(i) for i in data.split('-')]
    #temp = (request.args.get('temp'))
    # emg = float(request.args.get('emg'))
    #hrt = (request.args.get('hrt'))
    #acc = (request.args.get('acc'))
    #arr = [oxy, temp, hrt, acc]
    # pred = m.predict(arr)
    #arrs.append(arr)
    print(application2.logger.info(arr))
    print("Oximeter: {oxy},Temperature: {temp} , Heart rate: {hrt} ,EMG: {emg}, Activity: {acc}".format(oxy= arr[0] , temp =arr[1] , hrt=arr[2] , emg = 157 ,acc=arr[3]))
    X = np.array(arr)
    #X = X.astype(np.float)
    # X = StandardScaler().fit_transform(X)
    X = np.reshape(X, (1, 1, 4))
    p = m.predict(X)
    err = tf.keras.losses.mae(flatten(X), flatten(p))
    # print(np.shape(X))
    if err >= Threshold:

            response = requests.request("GET", url, headers=headers, params=querystring)
            print(response.text)
            print("Seizure is near")

            return ("Seizure is near!")

            # print(err)
    else:
            print("Normal")
            return ("Normal")



if __name__ == '__main__':
    application2.run()





#C:\Users\Administrator\.PyCharmCE2019.2\config\scratches
