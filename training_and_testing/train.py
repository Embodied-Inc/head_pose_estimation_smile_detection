import os
import sys
sys.path.append('..')
import logging
import argparse
import pandas as pd
import numpy as np

from lib.SSRNET_model import *

import TYY_callbacks
from TYY_generators import *

import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"], d["smile"]

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    parser.add_argument("--db_name", default='../data/train',
                        help="Path to train db name")
    parser.add_argument("--output", default='TRAINED',
                        help="Path to train db name")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    db_name = args.db_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    output = args.output 
    image_size = 64

    logging.debug("Loading data...")
    db_list = []
    for _,_,filenames in os.walk(db_name):
        for train_db in filenames:
            extension = os.path.splitext(train_db)[1]
            if extension == '.npz':
                db_list.append(os.path.join(db_name, train_db))
    image = []
    pose = []
    smile = []
    if not db_list:
        print('db_name is wrong!!!')
        return
    for i in range(0,len(db_list)):
        image_temp, pose_temp, smile_temp = load_data_npz(db_list[i])
        image.append(image_temp)
        pose.append(pose_temp)
        smile.append(smile_temp)
    image = np.concatenate(image,0)
    pose = np.concatenate(pose,0)
    smile = np.concatenate(smile,0)
        
    # we only care the angle between [-99,99] and filter other angles
    x_data = []
    y_pose = []
    y_smile = []
    print(image.shape)
    print(pose.shape)
    print(smile.shape)
    for i in range(0,pose.shape[0]):
        temp_pose = pose[i,:]
        if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
            x_data.append(image[i,:,:,:])
            y_pose.append(pose[i,:])
            y_smile.append(smile[i,:])
    
    x_data = np.array(x_data)
    y_pose = np.array(y_pose)
    y_smile = np.array(y_smile)

    start_decay_epoch = [30,60]

    optMethod = Adam()

    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = 3
    isFine = False
    model = SSR_net_MT(image_size, num_classes, stage_num, lambda_d)()
    save_name = 'ssrnet_mt'

    model.load_weights("TRAINED_models_tf2/ssrnet_mt/ssrnet_mt.h5")
    model.compile(optimizer=optMethod, loss=["mae", tf.losses.categorical_crossentropy], loss_weights=[0.1, 1])
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(output+"_models")
    mk_dir(output+"_models/"+save_name)
    mk_dir(output+"_models/"+output+"_checkpoints")
    plot_model(model, to_file=output+"_models/"+save_name+"/"+save_name+".png")
    for i_L,layer in enumerate(model.layers):
        if i_L >0 and i_L< len(model.layers)-1:
            if 'pred' not in layer.name and 'caps' != layer.name and 'merge' not in layer.name and 'model' in layer.name:
                plot_model(layer, to_file=output+"_models/"+save_name+"/"+layer.name+".png")
    

    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(output+"_models/"+output+"_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate]

    logging.debug("Running training...")
    
    data_num = x_data.shape[0]
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    x_data = x_data[indexes]
    y_smile = y_smile[indexes]
    y_pose = y_pose[indexes]
    train_num = int(data_num * (1 - validation_split))
    
    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_smile = y_smile[:train_num]
    y_test_smile = y_smile[train_num:]
    y_train_head = y_pose[:train_num]
    y_test_head = y_pose[train_num:]


    hist = model.fit_generator(generator=data_generator_pose(X=x_train, Y1=y_train_head, Y2 = y_train_smile, batch_size=batch_size),
                                       steps_per_epoch=train_num // batch_size,
                                       validation_data=(x_test, [y_test_head, y_test_smile]),
                                       epochs=nb_epochs, verbose=1,
                                       callbacks=callbacks)
    
    logging.debug("Saving weights...")
    print("________", os.path.join(output+"_models/"+save_name, save_name+'.h5'))
    model.save_weights(os.path.join(output+"_models/"+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(output+"_models/"+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()
