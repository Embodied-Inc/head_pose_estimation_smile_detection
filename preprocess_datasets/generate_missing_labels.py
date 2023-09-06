import os
import sys
sys.path.append('..')
import logging
import argparse
import cv2
import numpy as np
from keras.layers import *
from keras.utils import plot_model
import tensorflow as tf
#from lib.SSRNET_model_old import *


_IMAGE_SIZE = 64

def load_data_npz(npz_path):
    d = np.load(npz_path)
    if "smile" in d.files and "pose" not in d.files:
        print("SMILE")
        return d["image"], d["smile"], 1 
    elif "pose" in d.files and "smile" not in d.files:
        print("POSE")
        return d["image"], d["pose"], 0
    else:
        #return d["image"],d["pose"],d["smile"]
        print("THERE ALREADY ARE POSE AND SMILE")
        exit(0)

def get_args():
    parser = argparse.ArgumentParser(description="This script tests the model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, 
                        help="Path to model path")
    parser.add_argument("--test_db", required=True, default='../data/test/data_file_name.npz', 
                        help='Path to test dataset')
    
    parser.set_defaults(use_pretrained=False)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_path = args.model_path
    test_db = args.test_db
    test_db_list = []
    if not os.path.isfile(test_db):
        print("{0} is not a file.".format(test_db))
        exit(0)
    extension = os.path.splitext(test_db)[1]
    if extension != '.npz':
        print("{0} is not a .npz file.".format(test_db))
        exit(0)

    image, pose, is_smile = load_data_npz(test_db)
    if is_smile:
        out_imgs = []
        model_extension = os.path.splitext(model_path)[1]  
        if model_extension == '.tflite':
            print("TfLite model evaluation.")
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            p_data = []
            # Test model on random input data.
            for i in range(len(image)):
                gray = image[i]
                gray = gray.reshape(64,64,1)
                out_imgs.append(gray)
                input_data = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                input_data = np.expand_dims(input_data,axis=0)
                input_data = np.array(input_data, dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                p_data.append(list(output_data))
            p_data = np.vstack(p_data)
            output_name = os.path.splitext(test_db)[0].split("/")[-1]
            np.savez('../data/train/'+output_name,image=np.array(out_imgs), pose=np.array(p_data), smile = pose, img_size=64)
        else:
            print("Your model is not tflite.")
            exit(0)
    else:
        model_extension = os.path.splitext(model_path)[1]  
        em_data = []
        out_imgs = []
        if model_extension == '.tflite':
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            p_data = []
            # Test model on random input data.
            for i in range(len(image)):
                gray = image[i]
                gray = cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY)
                out = gray.reshape(64,64,1)
                out_imgs.append(out)
                input_data = gray.reshape((1, 64, 64, 1))
                input_data = np.array(input_data, dtype=np.float32)
                #input_data /= 255
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                p_data.append(list(output_data))
                if output_data[0][0] < output_data[0][1]:
                    emotion = np.array([0, 1])
                else:
                    emotion = np.array([1, 0])
                em_data.append(emotion)

            output_name = os.path.splitext(test_db)[0].split("/")[-1]
            np.savez('../data/train/'+output_name ,image=np.array(out_imgs), pose=pose, smile = em_data, img_size=64)

if __name__ == '__main__':    
    main()
