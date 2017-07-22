#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Alex (CC by 3.0) 2017
#
# This work is licensed under a Creative Commons Attribution-NonCommercial 3.0 
# Unported License. This means you're free to copy, share, and build on this 
# work, but not to sell it. 
#
# Based on the works of
#  . "A Neural Network in 11 lines of Python (Part 1)" (c) iamtrask
#  . Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015 
#  . "Machine Learning A-Z™: Hands-On Python & R In Data Science" Udemy Course by Kirill Eremenko and Hadelin de Ponteves
#
#
#
# The goal of this project is just to get confortable with all the maths behind 
# artificial neural networks.
#


import random
import numpy as np

#np.random.seed(123)

X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])

y_AND  = np.array([0, 0, 0, 1]).T
y_NAND = np.array([1, 1, 1, 0]).T
y_OR   = np.array([0, 1, 1, 1]).T
y_XOR  = np.array([0, 1, 1, 0]).T

        
class ANN(object):
    # toDO:
    # [] Minibatch
    # [] Shuffle X+y
    # [] Output loss
    # [] Dynamic Learning Rates (Ex Adam)
    
    
    def __init__(self, sizes):
        self.num_layers = sizes
        
    def sigmoid ( self, z):
        return 1. / (1 + np.exp(-z))

    def sigmoid_deriv( self, z):
        return self.sigmoid(z) * ( 1 - self.sigmoid(z))
    
    def cost_derivative( self, y_pred, y):
        # Derivate of the Cost Function MSE = 0.5 * ( y^ - y)**2 = (y^ - y)
        return (y_pred - y)
        
    
    def predict( self, x):
        activation = np.reshape( x, (len(x), 1))
        
        for w,b in zip( self.weights, self.biases):
            z = np.dot( w, activation) + b
            activation = self.sigmoid(z)

        return 1 if 0.5 < activation else 0
            
    
    def fit( self, X, y, eta = 0.2, batch_size = 1, nb_epoch = 1):
        self.biases  = [ np.random.randn( x,1) for x in self.num_layers[1:]]
        self.weights = [ np.random.randn( y,x) for x,y in zip(self.num_layers[:-1], self.num_layers[1:])]

        training = X
        for epoch in range(nb_epoch):
            
            for i in range(len(training)):
                # Backpropagation's Algorith
                
                # 1. Set the corresponding activation a1 for the input layer
                activation = training[i:i+1].T # (n,1)
                activations = [ activation] # list of all activations
                zs = [] # list of all z weighted sums

                # 2. For each layer compute zl and al
                for w,b in zip( self.weights, self.biases):
                    z = np.dot( w, activation) + b
                    zs.append(z)
                    
                    activation = self.sigmoid(z)
                    activations.append(activation)
                
                # 3. Compute Output Error delta
                delta = self.cost_derivative( activations[-1], y[i]) * self.sigmoid_deriv(zs[-1]) #BP1
                
                # Output Layer-L
                self.biases[-1]  -= eta * delta
                self.weights[-1] -= eta * np.dot( delta, activations[-2].T)
                                
                # 4. Backprogpagate the error to the hidden layers L-1, .. , 2
                ndelta = delta
                for l in range( 2, len(self.num_layers)):
                    ndelta = np.dot( self.weights[ -l+1].T, ndelta) * self.sigmoid_deriv(zs[-l])  #(n,1)

                    # 5. Update the weights and biases
                    self.biases[-l]  -= eta * ndelta
                    self.weights[-l] -= eta * np.dot( ndelta, activations[-l-1].T)
                    
        
        

model = ANN( [2, 4, 1])
model.fit( X, y_XOR, nb_epoch = 1500)

print("Predict (0,0): {}".format( model.predict( (0,0))))
print("Predict (0,1): {}".format( model.predict( (0,1))))
print("Predict (1,0): {}".format( model.predict( (1,0))))
print("Predict (1,1): {}".format( model.predict( (1,1))))
