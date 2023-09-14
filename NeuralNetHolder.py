
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random, exp, dot
from sklearn.model_selection import train_test_split
import pickle
from Lunar_Lander import NN

class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        with open("./nn_obj.pickle", "rb") as f:
            self.nn_obj = pickle.load(f) 

        self.max_from_data = [798.248414,788.964754, 7.993805, 5.990543]
        self.min_from_data = [ -745.898304,65.798294,-5.300000,-6.867767]
         
    def normalized(self, x,minimum, maximum):
        normalizedData =2*((x - minimum)/(maximum - minimum))
        return normalizedData
    
    def denormalized(self, x_norm, maximum, minimum):
        denormalized_val = x_norm * (maximum - minimum) + minimum
        return denormalized_val
    
    
    
   

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        lst = input_row.split(',')
        input_row = [float(i) for i in lst]
        x1_dist = int(self.normalized(input_row[0],self.min_from_data[0],self.max_from_data[0]))
        x2_dist= int(self.normalized(input_row[1],self.min_from_data[1],self.max_from_data[1]))
        input_row = np.array([[x1_dist,x2_dist]])
        prediction = self.nn_obj.forward_propagation(input_row)
        y1_velo = self.denormalized(prediction[0,0],self.max_from_data[2],self.min_from_data[2])
        #Y1 = prediction[0,0]
        #Y2 = self.denormalized(prediction[0,1],self.max_from_data[3],self.min_from_data[3])  
        y2_velo =prediction[0,1]
        
        return y1_velo,y2_velo
        
        
