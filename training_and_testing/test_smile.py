import os
import sys
sys.path.append('..')
import logging
import argparse

import cv2
import numpy as np
#import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
#from keras.utils import np_utils
#from keras.optimizers import SGD, Adam
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from lib.SSRNET_model import *

#import TYY_callbacks
#from TYY_generators import *

_IMAGE_SIZE = 64

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["smile"]

def get_args():
    parser = argparse.ArgumentParser(description="This script tests the model for head smile estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, 
                        help="Path to model path")
    parser.add_argument("--test_db", required=False, default='../data/smile_test', 
                        help='Path to test dataset')
    
    parser.set_defaults(use_pretrained=False)

    args = parser.parse_args()
    return args

def main():
    #K.clear_session()
    #K.set_learning_phase(0) # make sure its testing mode
    
    args = get_args()
    
    model_path = args.model_path
    test_db = args.test_db
    test_db_list = []
    for _, _, test_file_names in os.walk(test_db):
        for filename in test_file_names:
            extension = os.path.splitext(filename)[1]
            if extension == '.npz':
                test_db_list.append(os.path.join(test_db,filename))

    x_data = []
    y_data = []
    for test_db_name in test_db_list:
        image, smile = load_data_npz(test_db_name)
        print(smile.shape)
        for i in image:
            x_data.append(i)

        for i in smile:
            y_data.append(i)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(smile.shape)
       
    model_extension = os.path.splitext(model_path)[1]  
    if model_extension == '.h5':
        print("Keras model evaluation.")
        stage_num = [3,3,3]
        lambda_d = 1
        num_classes = 3
        model = SSR_net_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_mt'
        model.load_weights(model_path)
        p_head, p_data = model.predict(x_data)
        p_data = np.array(p_data)
    elif model_extension == '.tflite':
        print("TfLite model evaluation.")
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        p_data = []
        # Test model on random input data.
        for i in range(len(x_data)):
            input_data = x_data[i]
            input_data = np.reshape(input_data, (1,64,64,1))
            input_data = np.array(input_data, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            p_data.append(list(output_data[0]))
        p_data = np.vstack(p_data)
            
    else:
        print("Invalid extention of model.")
        exit(0) 
   
        #p_data = np.array(p_data)
    #print(p_data.shape)
    
    k  = 0
    for i in range(p_data.shape[0]):
        if p_data[i][0]>p_data[i][1] and y_data[i][0] > y_data[i][1]:
            k+=1
        if p_data[i][0]< p_data[i][1] and y_data[i][0] < y_data[i][1]:
            k+=1

    print(k/p_data.shape[0])
if __name__ == '__main__':    
    main()
