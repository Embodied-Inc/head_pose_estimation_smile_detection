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
    parser.add_argument("--db", type=str, default='./database',
                        help="path to database")
    parser.add_argument("--output", type=str, default='',
                        help="path to output database dir")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--face_detector", type=int, default=0,
                        help="1 for MTCNN face detector and 0 for our face detector.")
    parser.add_argument("--ad", type=float, default=0.4,
                        help="enlarge margin")
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
        mypath_obj = os.path.join(mypath, dir_name)
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
            img = cv2.imread(os.path.join(mypath_obj, img_name))
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

                        x1,y1,w,h = d['box']
                        x2 = x1 + w
                        y2 = y1 + h
                        xw1 = max(int(x1 - ad * w), 0)
                        yw1 = max(int(y1 - ad * h), 0)
                        xw2 = min(int(x2 + ad * w), img_w - 1)
                        yw2 = min(int(y2 + ad * h), img_h - 1)
                        break
                if is_detected:
                    img = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                    [xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [xw1,xw2,yw1,yw2]
                    if isPlot:
                        print([xw1_pre,xw2_pre,yw1_pre,yw2_pre])
                        cv2.imshow('check',img)
                        k=cv2.waitKey(10)
                    img = cv2.resize(img, (img_size, img_size))
                    cont_labels = np.array([yaw, pitch, roll])
                    out_imgs.append(img)
                    out_poses.append(cont_labels)
    
    c = list(zip(out_imgs, out_poses))
    random.shuffle(c)
    out_imgs = [e[0] for e in c]
    out_poses = [e[1] for e in c]
    
    index = int(0.7 * len(out_imgs))
    train_imgs = out_imgs[:index]
    train_poses = out_poses[:index]
    test_imgs = out_imgs[index:]
    test_poses = out_poses[index:]

    np.savez(output_path+'train',image=np.array(train_imgs), pose=np.array(train_poses), img_size=img_size)
    np.savez(output_path+'test',image=np.array(test_imgs), pose=np.array(test_poses), img_size=img_size)
    print("len : ", len(out_imgs))

if __name__ == '__main__':
	main()
