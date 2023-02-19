
'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from  sklearn.utils import shuffle
'''
import os
import pandas as pd
import numpy as np
import matplotlib.image as img
import cv2
from pathlib import Path
from numpy import savetxt

'''
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
'''

train1 = pd.read_csv("Sets/train1.csv")
train2 = pd.read_csv("Sets/train2.csv")
train3 = pd.read_csv("Sets/train3.csv")
train4 = pd.read_csv("Sets/train4.csv")
train5 = pd.read_csv("Sets/train5.csv")
f1 = pd.read_csv("Sets/validation1.csv")
f2 = pd.read_csv("Sets/validation2.csv")
f3 = pd.read_csv("Sets/validation3.csv")
f4 = pd.read_csv("Sets/validation4.csv")
f5 = pd.read_csv("Sets/validation5.csv")

#Remove extra column
f1 = f1.drop(columns = ["Unnamed: 0"])
f2 = f2.drop(columns = ["Unnamed: 0"])
f3 = f3.drop(columns = ["Unnamed: 0"])
f4 = f4.drop(columns = ["Unnamed: 0"])
f5 = f5.drop(columns = ["Unnamed: 0"])
train1 = train1.drop(columns = ["Unnamed: 0"])
train2 = train2.drop(columns = ["Unnamed: 0"])
train3 = train3.drop(columns = ["Unnamed: 0"])
train4 = train4.drop(columns = ["Unnamed: 0"])
train5 = train5.drop(columns = ["Unnamed: 0"])



#Remove extra row from train4 and add to f4
row = train2.sample(1).index
row_values = train2.iloc[row].to_numpy()
train2 = train2.drop(row)
f2.loc[len(f2.index)] = row_values[0]

#Remove extra row from train4 and add to f4
row = train5.sample(1).index
row_values = train5.iloc[row].to_numpy()
train5 = train5.drop(row)
f5.loc[len(f5.index)] = row_values[0]


training_sets = np.array([train1, train2, train3, train4, train5])
test_sets = np.array([f1, f2, f3, f4, f5])

train_TP = 0
train_NP = 0
train_FP = 0
train_FN = 0

test_TP = 0
test_NP = 0
test_FP = 0
test_FN = 0


final_train_w = np.array([])
final_train_err = np.array([])
final_test_w = np.array([])
final_test_err = np.array([])






#Function to calculate weights
def w(X,Y):
    X = X.astype('float32')
    A = np.linalg.pinv(np.dot(X.T,X))
    B = np.dot(X.T,Y)
    w = np.dot(A,B)
    return w
    
#Function to calculate the error term
def J(X, Y, w):
    predict = np.dot(X,w)
    A = predict.flatten() - Y
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(A)):
        
        yesNo = Y[i]
        if yesNo == 1:
            if A[i] <= 0:
                TP += 1
            if A[i] > 0.5:
                FN += 1
        if yesNo == 0:
            if A[i] < 0.5:
                TN += 1
            if A[i] >=0.5:
                FP += 1
            
    return TP,TN,FP,FN


def std(strin):
    img = cv2.imread(strin)
    height = img.shape[0]
    width = img.shape[1]
    index = height/2
    index1 = width/2
    #print(height,width)
    
    grayvals = np.array([])
    temparray = np.array([])
    
    for i in range(0,height):
        temparray = np.array([])
        for j in range(0,width):
            temparray = np.append(temparray,img[i,j][0])

        grayvals = np.concatenate((grayvals, temparray),axis=0)

    grayvals = grayvals.reshape(width, height)

    currClust=0
    clustOverall=np.array([])
    for yc in range(1,11):
        perch = (yc*(int(0.1*height)))
        for xc in range (1,11):
            percw = (xc*(int(0.1*width)))
            clust = np.array([])
            for y in range(perch-int(0.1*height),perch):
                temparray = np.array([])
                #print("Aaaaaaaaaaaaa: "+str(i))
                for x in range(percw-int(0.1*width),percw):
                   # print("x: "+str(x)+" y: "+str(y))
                    temparray = np.append(temparray, img[y,x][0])
                num = np.average(temparray)
                clust=np.append(clust, num)
            clustOverall = np.append(clustOverall, clust)
            
    std_d = np.std(clustOverall)
    r = np.max(clustOverall) - np.min(clustOverall)
    return std_d, r
    

for i in range(len(training_sets)):
    
    train_weights = np.array([[0]])
    test_weights = np.array([[0]])

    train = training_sets[i]
    test = test_sets[i]
    
    std_val = np.array([[0]])
    range_val = np.array([[0]])

    for j in range(len(train)):
        
        str_path = "brain_tumor/" + train[j][1]
        #str_path = 'brain_tumor/Y1.jpg' 
        #path = Path(str_path)
        #print(path)

        std_d, r = std(str_path)
        std_val = np. concatenate((std_val, np.array([[std_d]])), axis = 0)
        range_val = np.concatenate((range_val, np.array([[r]])), axis = 0)
   
    #
    range_val = np.delete(range_val, 0, 0)
    std_val = np.delete(std_val, 0, 0)
    
    train = np.append(train, range_val, axis = 1)
    train = np.append(train, std_val, axis = 1)
    
    X = train[:, [2, 3]]
    Y = train[:, 0]
    
    train_weight = w(X,Y)
    
    final_train_w = np.append(final_train_w, train_weight)
    train_wx = np.array([[train_weight[0]]])
    train_wy = np.array([[train_weight[1]]])
    
    train_weights = np.concatenate((train_weights, train_wx),axis = 0)
    train_weights = np.concatenate((train_weights, train_wy),axis = 0)
    train_weights = np.delete(train_weights, 0,0)
    
    train_TP, train_TN, train_FP, train_FN= J(X,Y,train_weights)
    
    accuracy = (train_TP+train_TN)/(train_TP+train_TN+train_FP+train_FN)
    prec = (train_TP)/(train_TP+train_FP)
    recall = (train_TP)/(train_TP+train_FN)
    f1 = 2 * (1/((1/prec)+(1/recall)))
    
    final_train_err = np.append(final_train_err, [train_TP, train_TN, train_FP, train_FN,
                                                  accuracy, prec, recall, f1])

    
    
    
    
    std_val = np.array([[0]])
    range_val = np.array([[0]])

    for j in range(len(test)):
        
        str_path = "brain_tumor/" + test[j][1]
        #str_path = 'brain_tumor/Y1.jpg' 
        #path = Path(str_path)
        #print(path)

        std_d, r = std(str_path)
        std_val = np. concatenate((std_val, np.array([[std_d]])), axis = 0)
        range_val = np.concatenate((range_val, np.array([[r]])), axis = 0)
   
    range_val = np.delete(range_val, 0, 0)
    std_val = np.delete(std_val, 0, 0)
    
    test = np.append(test, range_val, axis = 1)
    test = np.append(test, std_val, axis = 1)
    
    X = test[:, [2, 3]]
    Y = test[:, 0]
    
    
    test_TP, test_TN, test_FP, test_FN = J(X,Y, train_weights)
    
    accuracy = (test_TP+test_TN)/(test_TP+test_TN+test_FP+test_FN)
    prec = (test_TP)/(test_TP+test_FP)
    recall = (test_TP)/(test_TP+test_FN)
    f1 = 2 * (1/((1/prec)+(1/recall)))

    final_test_err = np.append(final_test_err, [test_TP, test_TN, test_FP, test_FN,
                                                accuracy,prec,recall,f1])
    
    
    
print("*************")
print(final_train_w)
print(final_train_err)
print(final_test_w)
print(final_test_err)

savetxt('Data/train_weights.csv', final_train_w, delimiter=',')
savetxt('Data/test_weights.csv', final_test_w, delimiter=',')
savetxt('Data/train_errors.csv', final_train_err, delimiter=',')
savetxt('Data/test_errors.csv', final_test_err, delimiter=',')
    


