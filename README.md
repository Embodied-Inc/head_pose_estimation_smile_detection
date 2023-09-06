# Head pose estimation

This project is intended to solve Head Pose Estimation issue. The goal of this project is to detect the position of the head using only the face image (without detecting landmarks). The position of the head can be described with the following angles:
*  Yaw
*  Pitch
*  Roll

We used **SSR Neural Network** to solve this issue.

## SSR Neural Network

The architecture of this network can be found [here](resources/ssrnet_mt.png).  
SSR-Net has been chosen to solve this issue because of the following advantages:
*  Doesn’t contain custom layers and can be easily converted to TFLite model.
*  SSR-Net’s performance approaches those of the state-of-the-art methods whose model sizes are larger.

## Dependencies

```
pip3 install requirments.txt
```

## Datasets
The following datasets are used
####  Training
*  300W-LP
*  BIWI_train
*  BP4D_train  

####  Testing
*  AFLW2000
*  BIWI_test
*  BP4D_test

You can find the data from the following [link](https://s3.console.aws.amazon.com/s3/object/head-pose-estimation-dataset/data.zip?region=eu-central-1&tab=overview).

## Codes

There are two different section of this project.
1.  Data pre-processing
2.  Training and testing

We will go through the details in the following sections.

### 1. Data pre-processing

####  BP4D dataset
Originally BP4D dataset doesn’t contain the labels to train NN for head pose estimation. The scripts have been implemented to generate the synthetic labeled dataset.
If you want to do the pre-processing from the beginning, you need to download the [BP4D](https://s3.console.aws.amazon.com/s3/object/head-pose-estimation-dataset/BP4D_data.zip?region=eu-central-1&tab=overview) dataset first. This contains the object files and the straight face images.
Put BP4D folder under **data/**

####  Run pre-processing  
For generating face images with different rotation angles you need to clone the [face3d](https://github.com/YadiraF/face3d) module, follow the given instructions in that module and put it under the **/head_pose_estimation**.
```
cd preprocess_datasets
python3 head_pose_transform.py --image_dir ../data/BP4D/imgs --obj_dir ../data/BP4D/objs --output_directory <path/to/output/dir>
python3 generate_npz.py --db <path/to/output/dir>
```
Put generated **train.npz** and **test.npz** files under **../data/train/** and **../data/test/** respectivly.

### 2. Training and testing 
```
# Training

cd training_and_testing
sh run_train.sh

# Testing

cd training_and_testing
python3 test.py --model_path <path/to/model> --test_db <path/to/test/db>
```

### Conversion to tflite model
```
python3 convert_h5_to_tflite.py --h5_path <path/to/keras/model> --output <path/to/tflite/model>
```





