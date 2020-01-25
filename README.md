# digit-recognizer-web-app

## Overview

Flask web app that recognizes handwritten digits drawn in the browser using convnets. Here is a screenshot of the final result:

![Alt text](screenshot.png?raw=true "screenshot")

* First, we build a model that recognizes handwritten digit images trained on MNIST dataset developped using [Keras](http://keras.io/) on top of [TensorFlow](https://www.tensorflow.org/) backend. 
* Second, we save the resulting weights to the model directory. 
* Third, the code is wrapped into a Webapp using [Flask](http://flask.pocoo.org/) Micro Framework to serve the pretrained models in order to recognize the drawn digit.

## Dependencies

```sudo pip install -r requirements.txt```

## Usage

Once dependencies are installed, run this to see it in your browser. 

```python app.py```

