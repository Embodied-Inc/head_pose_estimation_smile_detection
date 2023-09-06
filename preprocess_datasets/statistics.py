import scipy.io as sio
import random
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
from moviepy.editor import *
import numpy as np
import argparse
from mtcnn.mtcnn import MTCNN
from face_detector import FaceDetector 


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, required = True,
                        help="path to database")
    parser.add_argument("--output", type=str, default='output',
                        help="path to output database dir")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--face_detector", type=int, default=0,
                        help="1 for MTCNN face detector and 0 for our face detector.")
    parser.add_argument("--ad", type=float, default=0.4,
                        help="enlarge margin")
    parser.add_argument("--abs", type=int, default=0,
                        help="abs")
    parser.add_argument("--isPlot", type=bool, default=0,
                        help="plot cropped face images.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mypath = args.db
    output_path = args.output
    img_size = args.img_size
    ad = args.ad
    isPlot = args.isPlot
    if args.face_detector == 0:
        detector = FaceDetector()
    else:
        detector = MTCNN()
    
    onlyfiles_png = []
    onlyfiles_txt = []

    dir_names = []
    for _,dirs,_ in os.walk(mypath):
        for d in dirs:
            dir_names.append(d)
    for dir_name in dir_names:
        mypath_obj = os.path.join(mypath,dir_name)
        onlyfiles_txt_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.txt')]
        onlyfiles_png_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.png')]
        onlyfiles_txt_temp.sort()
        onlyfiles_png_temp.sort()
        onlyfiles_txt.append(onlyfiles_txt_temp)
        onlyfiles_png.append(onlyfiles_png_temp)
    print(len(onlyfiles_txt))
    print(len(onlyfiles_png))
    out_imgs = []
    out_poses = []
    for i in range(len(onlyfiles_png)):
        mypath_obj = os.path.join(mypath, dir_names[i])
        for j in tqdm(range(len(onlyfiles_png[i]))):
            img_name = onlyfiles_png[i][j]
            txt_name = onlyfiles_txt[i][j]
            img_name_split = img_name.split('_')
            txt_name_split = txt_name.split('_')
            if img_name_split[1] != txt_name_split[1]:
                print('Mismatched!')
                sys.exit()
            #pose_path = mypath_obj+'/'+txt_name
            pose_path = os.path.join(mypath_obj, txt_name)
            # Load pose in degrees
            pose_annot = open(pose_path, 'r')
            R = []
            for line in pose_annot:
                line = line.strip('\n').split(' ')
                L = []
                if line[0] != '':
                    for nb in line:
                        if nb == '':
                            continue
                        L.append(float(nb))
                    R.append(L)
            R = np.array(R);yaw = R[0][0];pitch = R[0][1]; roll = R[0][2]
            img = cv2.imread(mypath_obj+'/'+img_name)
            img_h = img.shape[0]
            img_w = img.shape[1]
            if j==0:
            	[xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [0,0,0,0]
            	[xw1,xw2,yw1,yw2] = [0,0,0,0]
            detected = detector.detect_faces(img)
            is_detected = False
            if len(detected) > 0:
                for i_d, d in enumerate(detected):
                    if d['confidence'] > 0.4:
                        is_detected = True
                if is_detected:
                    cont_labels = np.array([yaw, pitch, roll])
                    out_poses.append(cont_labels)
    
    print("len : ", len(out_poses))
    yaw_45 = 0
    yaw_60 = 0
    yaw_90 = 0
    yaw_n_45 = 0
    yaw_n_60 = 0
    yaw_n_90 = 0
    pitch_30 = 0
    pitch_45 = 0
    roll_30 = 0
    roll_45 = 0

    for i in out_poses:
        if args.abs:
            if abs(i[0]) <= 45:
                yaw_45 += 1
            elif abs(i[0]) <= 60 :
                yaw_60 += 1
            else:
                yaw_90 += 1
        
        else: 
            if i[0] < -60:
                yaw_n_90 += 1
            elif i[0] < -45:
                yaw_n_60 += 1
            elif i[0] < 0:
                yaw_n_45 += 1
            elif i[0] <= 45:
                yaw_45 += 1
            elif i[0] <=60:
                yaw_60 += 1
            else:
                yaw_90 += 1
                
        if abs(i[1]) <= 30:
            pitch_30 += 1
        else:
            pitch_45 +=1

        if abs(i[2]) <= 30:
            roll_30 += 1
        else: roll_45 += 1
    
    print("--------------Yaw-----------")
    print("Yaw    0:45  ", yaw_45)
    print("Yaw   45:60  ", yaw_60)
    print("Yaw   60:90  ", yaw_90)
    print("Yaw  -45:0   ", yaw_n_45)
    print("Yaw  -60:-45 ", yaw_n_60)
    print("Yaw  -90:-60 ", yaw_n_90)
    print("--------------Pitch-----------")
    print("Pitch <= 30: ", pitch_30)
    print("Pitch <= 45: ", pitch_45)
    print("--------------Roll-----------")
    print("Yaw <= 30: ", roll_30)
    print("Yaw <= 45: ", roll_45)

if __name__ == '__main__':
	main()
