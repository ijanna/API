from flask import Flask, request, jsonify, render_template
import werkzeug
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np
import pickle
import sklearn

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if(request.method == 'POST'):

        # Request
        imagefile = request.files['image']
        
        #read image file string data
        filestr = imagefile.read()

        #convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)

        # convert numpy array to image
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

        # Predict emotions with DeepFace
        predictions = DeepFace.analyze(img, actions = ['emotion'])
        emotion_dict = predictions['emotion']
 
        # Import the model
        with open('model/finalised_model.sav','rb') as f:
            model = pickle.load(f)

        # Preparing dataset received from DeepFace
        data = emotion_dict.values()
        r=[]
        for value in data:
            r.append(float(value))

        # Make Prediction
        status = model.predict([r])

        # Prepare prediction data to encode
        statusElement = status[0]
        statusStr = str(statusElement) 

        # Encode Response
        response = jsonify({
            "status" : statusStr
        })

        # Print in Debug mode
        print(statusStr)

        # Return reponse
        return response


if __name__ == '__main__':
    app.run(debug = True)