import os
import argparse
from lib.SSRNET_model import *
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="This script converts keras model to tflite.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5_path", required=True,help="Path to h5 model")
    parser.add_argument("--output", required=True,help="Output name")
    args = parser.parse_args()
    model = SSR_net_MT(64, 3, [3,3,3],1)()
    try:
        model.load_weights(args.h5_path)
    except:
        print("Not proper h5 file")
    converter = tf.lite.TFLiteConverter.from_keras_model(model) # Your model's name
    model = converter.convert()
    file = open( args.output , 'wb' )
    file.write( model )
if __name__ == '__main__':
    main()
