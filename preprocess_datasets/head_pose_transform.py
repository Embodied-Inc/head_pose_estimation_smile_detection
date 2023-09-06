#!/usr/bin/env python3
import os, sys
import numpy as np
import math
import scipy.io as sio
from skimage import io
from time import time
import subprocess
import pywavefront
import argparse
import time
import cv2

sys.path.append("../face3d")
import face3d
from face3d import mesh, mesh_numpy


def transform_test(vertices,triangles,texture,texture_coords, obj, camera, h,  w):
    '''
    Aegs:
        vertices: 3D model vertices
        triangles: 3D model triangles
        texture: texture specified in .mtl file (2D image)
        texture_coords: coords to  project a 2D image to a 3D model's surface
        obj: dict contains obj transform paras
	camera: dict contains camera paras
        h: generated 2D image height
        w: generated 2D image width

    Returns:
        generated 2D image
    '''

    R = mesh.transform.angle2matrix(obj['angles'])
    transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
    
    if camera['proj_type'] == 'orthographic':
        projected_vertices = transformed_vertices
        image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    else:
    
    	## world space to camera space. (Look at camera.) 
        camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
    	## camera space to image space. (Projection) if orth project, omit
        projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
    	## to image coords(position in image)
        image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

    tex_h, tex_w, _ = texture.shape
    
    #transform texture coordinates to fit in 3D model
    texcoord = np.zeros_like(texture_coords)
    texcoord[:,0] = texture_coords[:,0]*(tex_w - 1)
    texcoord[:,1] = texture_coords[:,1]*(tex_h - 1)
    texcoord[:,1] = tex_h - texcoord[:,1] - 1
    texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1)))) # add z 

    rendering = mesh.render.render_texture(image_vertices,triangles,texture,texcoord ,triangles, h, w, c = 3, mapping_type = 'nearest')
    rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering

def transform_image(image_path, image_base_name ,obj_path):
    
    start  = time.time()
    if not os.path.exists(os.path.join(args.output_directory, image_base_name)):
        os.mkdir(os.path.join(args.output_directory, image_base_name))
        scene = pywavefront.Wavefront(obj_path, create_materials=True, collect_faces=True)
        texture = io.imread(image_path)/255.
        vertices = scene.vertices
        triangles = np.array(scene.mesh_list[0].faces)
        texture_coords = np.array(scene.parser.tex_coords)
        vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
        obj,camera = {}, {}
        h, w = texture.shape[0], texture.shape[1]
        scale = min(h, w) * 0.85
        scale_init = scale/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
        camera['proj_type'] = 'orthographic'
         # 0 for yaw, 1 for roll ad 2 for pitch
        obj['s'] = scale_init
        obj['angles'] = [0, 0, 0]
        obj['t'] = [0, 0, 0]
        for pitch in [-45, -30, 0 , 30, 45]:
            for yaw in range(-90, 100, 10):
                for roll in [-45, -30, 0 , 30, 45]:
                    obj['angles'] = [pitch, yaw, roll]
                    image = transform_test(vertices,triangles,texture,texture_coords, obj, camera, h, w)
                    image = cv2.resize(image, (1100,900))
                    yaw *= -1
                    pitch*= -1
                    roll *= -1
                    image_name = image_base_name + "_" + str(yaw) + "_" +str(pitch) + "_" + str(roll)
                    io.imsave(os.path.join(args.output_directory, image_base_name, image_name + ".png"), image)
                    f = open(os.path.join(args.output_directory, image_base_name, image_name + ".txt") , "w")
                    f.write(str(yaw) + " " + str(pitch) + " " + str(roll) + "\n")
                    f.close()
                    start = time.time()
def get_args():
    parser = argparse.ArgumentParser(description="This script generates new images with different posses "
                                                 "and as input takes object files and corresponding straight face images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-img", "--image_dir", type=str, required=True, 
                                    help="Location of images.")
    parser.add_argument("-obj", "--obj_dir", type=str, required=True, 
                                    help="Location of objs.")
    parser.add_argument("-out", "--output_directory", type=str, required=True, 
                                    help="Location for storing generated images.")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.output_directory):
        try:
            os.mkdir(args.output_directory)
        except OSError:
            print ("Creation of the directory %s failed" % args.output_directory)
    for image in os.listdir(args.image_dir):
        image_name = image[:-4]
        real_image_directory = args.image_dir
        obj_directory = args.obj_dir
        transform_image(real_image_directory+"/"+image_name+".png", image_name, obj_directory +"/"+image_name+".obj")
