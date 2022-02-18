from cProfile import label
import numpy as np
# An example dataset generator
#from sklearn.datasets import make_blobs
import csv
from csv import reader, writer
import pandas as pd
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
    lamda = .01
    LCL = 0
    while (LCL - (lamda * (np.linalg.norm(np.array(weighted_vec)) **2)) ) > .001  : 
    
    for x in range(5):
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
            difference = [element * learning_rate for element in difference]
            #difference = learning rate* ((y-p)x - 2(lamda * w))
            temp = []
            for x in range(len(difference)):
                temp.append(weighted_vec[x] + difference[x])
            weighted_vec = temp
        
    accuracy = 0
    #Prediction
    with open(test_data_file) as f1, open(pred_file) as f2:
        csv_reader = reader(f1)
        csv_reader2 = writer(f2)
        count = 0
        for row in csv_reader:
            int_map = map(int, row)
            int_row = list(int_map)
            prediction = sigmoid(np.dot(weighted_vec, int_row))
            if prediction > .5:
                csv_reader2.writerow(["0"])
                if data[count][1] == 1:
                    accuracy += 1
            else:
                csv_reader2.writerow(["1"])
                if data[count][1] == 0:
                    accuracy += 0
            count += 1
    print(accuracy/3000)
            







        
        
    
    
  

if __name__ == "__main__":
    Xtrain_file = '../reference/Xtrain.csv'
    Ytrain_file = '../reference/Ytrain.csv'
    test_data_file = '../reference/Xtest.csv' 

    """ Xtrain_file = 'Xtrain.csv'
    Ytrain_file = 'Ytrain.csv'
    test_data_file = 'Xtrain.csv'  """


    pred_file = 'predictions.csv'
    run(Xtrain_file, Ytrain_file, test_data_file, pred_file)