import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

# Setting random seeds to keep everything deterministic.
seed = 1618
random.seed(seed)
np.random.seed(seed)
#tf.set_random_seed(seed)   # Uncomment for TF1.
tf.random.set_seed(seed)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#LGORITHM = "tf_net"

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)     #inputSize * neuronsPerLayer sized array with random weights
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    # Activation prime function.
    def sigmoidDerivative(self, x):
        sigX = self.sigmoid(x)
        return (sigX * (1 - sigX))

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #for kek, keko in zip(self.__batchGenerator(xVals, mbs), self.__batchGenerator(yVals, mbs)):
        #    print("Shape of X batch: %s" % str(kek.shape))
        #    print("Shape of Y batch: %s" % str(keko.shape))
        #i = 0
        for e in range(0, epochs):
            for xVal, yVal in zip(self.__batchGenerator(xVals, mbs), self.__batchGenerator(yVals, mbs)):
                self.__backprop(xVal, yVal)
                #i = i + 1
                #print("Batch no: %d done" % i)
                #print("Trained %d" % i)
            print("Epoch %d done training" % e)
        print("Training done :P")
        return

    def __backprop(self, inputX, inputY):   #inputX = (mbs, 784), inputY = (mbs, 10)
        layer1, layer2 = self.__forward(inputX)
        #layer1 = (mbs, 512)
        #layer2 = (mbs, 10)
        #Using MSE as cost function; MSE = 0.5*(out - expected)^2
        #print("Shape of layer1: %s" % str(layer1.shape))
        #print("Shape of layer2: %s" % str(layer2.shape))
        cost = layer2 - inputY   # (mbs, 10)
        #print("Shape of cost: %s" % str(cost.shape))
        derivativeA = layer2 * (1 - layer2) #(mbs, 10)
        #print("Shape of derivativeA: %s" % str(derivativeA.shape))
        deltaA = cost * derivativeA #(mbs, 10)
        #print("Shape of deltaA: %s" % str(deltaA.shape))
        layer1Adjust = np.matmul(deltaA, self.W2.transpose())  #(mbs, 512)
        #print("Shape of layer1Adjust: %s" % str(layer1Adjust.shape))
        derivativeB = layer1 * (1 - layer1) #(mbs, 512)
        #print("Shape of derivativeB: %s" % str(derivativeB.shape))
        deltaB = layer1Adjust * derivativeB #(mbs, 512)
        #print("Shape of deltaB: %s" % str(deltaB.shape))
        layer1sensitivity = np.matmul(inputX.transpose(), deltaB)   #(784, 512)
        #print("Shape of layer1sensitivity: %s" % str(layer1sensitivity.shape))
        layer2sensitivity = np.matmul(layer1.transpose(), deltaA)   #(512, 10)
        #print("Shape of layer2sensitivity: %s" % str(layer2sensitivity.shape))

        self.W2 = self.W2 - (self.lr * layer2sensitivity)
        #print("Shape of W2: %s" % str(self.W2.shape))
        self.W1 = self.W1 - (self.lr * layer1sensitivity)
        #print("Shape of W1: %s" % str(self.W1.shape))
        return

    # Forward pass.
    def __forward(self, input):
        layer1 = self.sigmoid(np.dot(input, self.W1))
        layer2 = self.sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []    
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    # we need to flatten the data first...
    xTrainFlat = xTrain.reshape(len(xTrain), 28*28)
    xTestFlat = xTest.reshape(len(xTest), 28*28)

    xTrain = xTrainFlat.astype('float64')   #Change type to float before trying to reduce range and copy over
    xTest = xTestFlat.astype('float64')

    xTrain = xTrain/255   #This works REALLY fast because of numpy
    xTest = xTest/255   #This works REALLY fast because of numpy

    print("Max xTrain: %f" % max(xTrain[0]))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("Type of xTrain dataset: %s." % str(xTrain.dtype))
    print("Type of xTest dataset: %s." % str(xTest.dtype))
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        myObj = NeuralNetwork_2Layer(784, 10, 512, 0.01)
        myObj.train(xTrain, yTrain, 40, True, 250)  #40, 250
        return myObj
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        lossType = keras.losses.mean_squared_error
        model = keras.Sequential()
        model.add(keras.Input(shape=28*28))
        model.add(keras.layers.Dense(512, activation=tf.nn.sigmoid))
        model.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))
        model.compile(optimizer='adam', loss=lossType, metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=5)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        pred = model.predict(data)
        return pred
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        pred = model.predict(data)
        return pred
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    #print("preds.shape = %s" % str(preds.shape))

    confusionMatrix = np.zeros((10, 10))

    for i in range(preds.shape[0]):
        #if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        indexPredicted = np.argmax(preds[i])
        indexActual = np.argmax(yTest[i])
        confusionMatrix[indexPredicted, indexActual] += 1
        if indexPredicted == indexActual:
            acc = acc + 1

    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))

    print("Confusion Matrix: ")
    print(confusionMatrix.astype(int))

    f1ScoreTotal = 0
    f1Scores = np.zeros(10)
    for i in range(0, 10):
        precision = confusionMatrix[i, i] / (np.sum(confusionMatrix[i,:]))
        if math.isnan(precision):
            precision = 0
        recall = confusionMatrix[i, i] / (np.sum(confusionMatrix[:, i]))
        if math.isnan(recall):
            recall = 1
        f1Scores[i] = 2 * ((precision * recall) / (precision + recall))
        if math.isnan(f1Scores[i]):
            f1Scores[i] = 0
        print("F-1 Score for %d is %f" % (i, f1Scores[i]))
        #print("\tPrecision: %f, Recall: %f" % (precision, recall))
        #f1ScoreTotal = f1ScoreTotal + f1Scores[i]

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

if __name__ == '__main__':
    main()
