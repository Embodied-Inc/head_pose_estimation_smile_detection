import os
import sys
sys.path.append('..')
import logging
import argparse

import cv2
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

from lib.SSRNET_model import *

_IMAGE_SIZE = 64

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]

def get_args():
    parser = argparse.ArgumentParser(description="This script tests the model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, 
                        help="Path to model path")
    parser.add_argument("--test_db", required=False, default='../data/test', 
                        help='Path to test dataset')
    
    parser.set_defaults(use_pretrained=False)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    model_path = args.model_path
    test_db = args.test_db
    test_db_list = []
    if test_db.endswith(".npz"):
         test_db_list = [test_db]

    else:
        for _, _, test_file_names in os.walk(test_db):
            for filename in test_file_names:
                extension = os.path.splitext(filename)[1]
                if extension == '.npz':
                    test_db_list.append(os.path.join(test_db,filename))

    for test_db_name in test_db_list:
        print("test_db_name: ", test_db_name) 
        image, pose = load_data_npz(test_db_name)
         # we only care the angle between [-99,99] and filter other angles
        x_data = []
        y_data = []

        for i in range(0,pose.shape[0]):
            temp_pose = pose[i,:]
            if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
                img = cv2.cvtColor(image[i,:,:,:], cv2.COLOR_BGR2GRAY)
                x_data.append(np.reshape(img,(64,64,1)))
                y_data.append(pose[i,:])
        

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print(x_data.shape)
        print(y_data.shape)
       
        model_extension = os.path.splitext(model_path)[1]  
        if model_extension == '.h5':
            print("Keras model evaluation.")
            stage_num = [3,3,3]
            lambda_d = 1
            num_classes = 3
            model = SSR_net_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
            save_name = 'ssrnet_mt'
            model.load_weights(model_path)
            p_data, p_smile = model.predict(x_data)
        
        elif model_extension == '.tflite':
            print("TfLite model evaluation.")
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            p_data = []
            # Test model on random input data.
            for i in range(x_data.shape[0]):
                input_data = x_data[i]
                input_data = np.expand_dims(input_data,axis=0)
                input_data = np.array(input_data, dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[1]['index'])
                p_data.append(list(output_data))
            p_data = np.vstack(p_data)
        else:
            print("Invalid extention of model.")
            exit(0) 
        
        pose_matrix = np.mean(np.abs(p_data-y_data),axis=0)
        MAE = np.mean(pose_matrix)
        yaw = pose_matrix[0]
        pitch = pose_matrix[1]
        roll = pose_matrix[2]
        print('\n--------------------------------------------------------------------------------')
        print(test_db_name+', ' + '(' +model_path+ ')' + ', MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll))
        #print(save_name+', '+test_db_name+'('+train_db_name+')'+', MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll))
        print('--------------------------------------------------------------------------------')

if __name__ == '__main__':    
    main()
