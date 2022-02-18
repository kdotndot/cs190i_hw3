from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
# An example dataset generator
#from sklearn.datasets import make_blobs
import csv
from csv import reader
import pandas as pd
import math
import random
from sklearn.datasets import make_classification

def sigmoid(z):
    #print(1.0/(1 + np.exp(-z)))
    return 1.0/(1 + np.exp(-z))



def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
	
    data = [] #Structure of data, data[0][0] is feature vector of first item, data[0][1] is label of first item
    
    with open(Xtrain_file) as f1, open(Ytrain_file) as f2:
        csv_reader = reader(f1)
        csv_reader2 = reader(f2)
    
        for row in csv_reader:
            
            int_map = map(int, row)
            int_row = list(int_map)
            data.append([int_row])
        
        count = 0
        for row in csv_reader2:
            if row[0] == "1":
                data[count].append(1)
                
            if row[0] == "0":
                data[count].append(0)

            count += 1
    
    weighted_vec = [0]*3000
    learning_rate = .01
    lamda = 1
    LCL = 0
    while (LCL - (lamda * (np.linalg.norm(np.array(weighted_vec)) **2)) ) > .001  : 
        shuffled = random.sample(data, len(data))
        for x in shuffled:
            #w = w + alpha( (y-p)x - 2(lamda * w))
            #left = (y-p)x
            #right = 2(lamda * w)
            
            x_vec = x[0]
            y = x[1]
            p = sigmoid(np.dot(weighted_vec, x[0]))
            left = [element * (y-p) for element in x_vec]
            right = [element * (2 * lamda) for element in weighted_vec]
            difference = []
            for x in range(0, len(left)):
                difference.append(left[x] - right[x])
            difference = [element * lamda for element in difference]
            #difference = (y-p)x - 2(lamda * w)
            temp = []
            for x in range(len(difference)):
                temp.append(weighted_vec[x] + difference[x])
            weighted_vec = temp
            
    print(weighted_vec)
        
        
    
    
  

if __name__ == "__main__":
    """ Xtrain_file = '../reference/Xtrain.csv'
    Ytrain_file = '../reference/Ytrain.csv'
    test_data_file = '../reference/Xtest.csv'  """

    Xtrain_file = 'Xtrain.csv'
    Ytrain_file = 'Ytrain.csv'
    test_data_file = 'Xtest.csv' 


    pred_file = 'predictions.csv'
    run(Xtrain_file, Ytrain_file, test_data_file, pred_file)