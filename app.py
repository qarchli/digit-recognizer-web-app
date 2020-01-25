# web app entry point
# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
# keras and handling images
from keras.models import model_from_json
from scipy.misc import imread, imresize, imshow
import tensorflow as tf
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
# for matrix math
import numpy as np
# for importing our keras model
import keras.models
# for regular expressions, saves time dealing with string data
import re
import base64
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
# ys.path.append(os.path.abspath("./model"))
# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
# the keras model and the computational graph
global model, graph


def init():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load woeights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model, graph


# initialize these variables
model, graph = init()

# decoding an image from base64 into raw representation


def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    # print(imgstr)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification

    # == Get the input image and transform it as the model expects ===
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    print("debug")
    # read the image into memory
    x = imread('output.png', mode='L')
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    # make it the right size
    x = imresize(x, (28, 28))
    # imshow(x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    print("debug2")

    # == Feed the final image to the model and predict ==
    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print("debug3")
        # convert the response to a string
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
    # optional if we want to run in debugging mode
    # app.run(debug=True)
