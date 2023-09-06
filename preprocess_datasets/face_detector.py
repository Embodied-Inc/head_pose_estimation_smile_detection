import numpy as np
import tensorflow as tf
import cv2
import os

#HOME_PATH = os.getenv("HOME")
#MODEL_PATH = HOME_PATH + "/workspace/cleaned_fsa/preprocess_datasets/face_detector_v1.tflite"
MODEL_PATH = "face_detector_model/face_detector_v1.tflite"

class FaceDetector:
    def __init__(self):
        # create tflite interpreter and load model to memory
        self.model_ = MODEL_PATH
        self.interpreter_ = tf.lite.Interpreter(self.model_)
        self.interpreter_.allocate_tensors()
        self.input_details = self.interpreter_.get_input_details()
        self.output_details = self.interpreter_.get_output_details()

    def preprocess_input(self, img):
        temp = cv2.resize(img, (320, 180), 0, 0, cv2.INTER_LINEAR)
        rgb_im = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        cv2.convertScaleAbs(rgb_im, rgb_im, 1.0/127.5)
        input_data = np.array(rgb_im, dtype=np.float32)
        input_data = input_data - np.array([1.0, 1.0, 1.0], dtype=np.float32)
        input_data = np.expand_dims(input_data, 0)
        self.interpreter_.set_tensor(self.input_details[0]['index'], input_data)

    def detect_faces(self, img):
        self.preprocess_input(img)
        detection = []
        self.interpreter_.invoke()
        rects = []
        scores = []
        det_bbox = self.interpreter_.get_tensor(self.output_details[0]['index'])
        det_scores = self.interpreter_.get_tensor(self.output_details[2]['index'])
        det_num = self.interpreter_.get_tensor(self.output_details[3]['index'])
        det_bbox = det_bbox[0]
        for i in range(int(det_num[0])):
            score = det_scores[0][i]
            rect = (int(img.shape[1]*det_bbox[i][1]), int(img.shape[0]*det_bbox[i][0]), int(img.shape[1]*(det_bbox[i][3] - det_bbox[i][1])),int(img.shape[0]*(det_bbox[i][2] - det_bbox[i][0])))
            detection.append({'confidence':score, 'box':rect})
        
        return detection
        
## preprocess input image
#img = cv2.imread("test.jpg") 
#temp = cv2.resize(img, (320, 180), 0, 0, cv2.INTER_LINEAR)
#rgb_im = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
#cv2.convertScaleAbs(rgb_im, rgb_im, 1.0/127.5)
#input_data = np.array(rgb_im, dtype=np.float32)
#input_data = input_data - np.array([1.0, 1.0, 1.0], dtype=np.float32)
#input_data = np.expand_dims(input_data, 0)
#
## pass image through network
#interpreter.set_tensor(input_details[0]['index'], input_data)
#interpreter.invoke()
#
## get output
#rects = []
#scores = []
#det_bbox = interpreter.get_tensor(output_details[0]['index'])
#det_scores = interpreter.get_tensor(output_details[2]['index'])
#det_num = interpreter.get_tensor(output_details[3]['index'])
#det_bbox = det_bbox[0]
#for i in range(int(det_num[0])):
#    score = det_scores[0][i]
#    scores.append(score)
#    rect = (det_bbox[i][1], det_bbox[i][0], det_bbox[i][3] - det_bbox[i][1], det_bbox[i][2] - det_bbox[i][0])
#    rects.append(rect)
#
#if 0.4 < scores[0]:
#    roi = (int(img.shape[1]*rects[0][0]), int(img.shape[0]*rects[0][1]), int(img.shape[1]*rects[0][2]), int(img.shape[0]*rects[0][3]))
#    cv2.rectangle(img, roi, (0, 255, 0))
#    cv2.imshow("Face", img)
#    cv2.waitKey(10000)
#else:
#    print("No faces detected")

