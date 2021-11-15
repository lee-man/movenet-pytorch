# Movenet Pytorch

This repository contains a PyTorch implementation of [Movenet](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) from Google. The inference correctness is the point this repo focuses on. If you are interested in Movenet training/finetuning, you can refer to my another repo [lee-man/movenet](https://github.com/lee-man/movenet).

### Credits

The original model, weights, etc. was created by Google and can be found at [tf hub](https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3).


Portions of the code in this repo are borrowed from the following repos:
1. [Centernet](https://github.com/xingyizhou/CenterNet) for prediction heads.
2. [Pytorch Vision](https://github.com/pytorch/vision) for MobileNet and Feature Pyramid Network.
3. [posenet-pytorch](https://github.com/rwightman/posenet-pytorch) for the other utility functions.
4. [TF official tutorial for Movenet](https://www.tensorflow.org/hub/tutorials/movenet) for image drawing.
5. [TensorFlow Lite Pose Estimation Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android) for Android deployment.


### Notice
In order to get in touch with the internal computational flow of Movenet, I use [netron](https://github.com/lutzroeder/netron) to visualize Movenet and my own implementation. You can use netron to compare the one with the other. The model definitions in onnx format or tflite format are maintained in `_models` directory. There are a few things to pay attention to when I convert the movenet TFLite model to PyTorch model:

* Movenet version:
  * There are two categories of Movenet: Lightning and Thunder. The current code supports Lightning version 3 and Thunder version 3. Google haven't released the full-precision version of Lightning version 4. The version 4 model should have improved prediction accuracy as synthetic images with under-represented poses are inlcluded in the training dataset.
* Extract the weights:
  * Currently I use the most clumsy way to extract the weights from TFLite model: open it with netron, select the layers/ops I want to deal with, export the weights, and rename the numpy file using PyTorch convention.
  * Once you extract all the weights, place them under `_models\weights` directory. `movenet.pth` will be generated under `_models` directory when you first run the program. 
* Batchnorm:
  * As the TFLite fuses the convolutional layers and batchnorm layers together, the BatchNorm layers are removed from PyTorch MobileNet v2 implementation.
* Padding:
  * The ways how Tensorflow and PyTorch do the padding are slightly different. In Tensorflow and original Movenet, the padding mode is specified as `same`, and it will do the asymmetry padding with the setting: `stride=2, kernel_size=3`. But in PyTorch, the padding will be symmetry. So I add `nn.ZeroPad2d` layers before Conv2d layer with `stride=2, kernel_size=3`.
* Feature Pyramid Network:
  * The structure of FPN used in Movenet is different from the official implementation in Pytorch Vision. In Movenet, the size of feature map for top-donwn structure of FPN is incrementally increased.
  * ~~The upsampling mechanism used in Movenet is `bilinear` interpolatation with `align_corners=False, half_pixel_centers=True`. It's fine to use this setting under PyTorch inference. But it seems that ONNX doesn't support this kind of operation yet, so I change it to `nearest` mode when I convert the model to ONNX format.~~

### Install

A suitable Python 3.x environment with a recent version of PyTorch is required. Development and testing was done with Python 3.8.8 and PyTorch 1.9.0 w/o GPU from Conda.


### Usage

There are three demo apps and one notebook in the root that utilize the Movenet model. 

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

From my benchmarking results, the current Pytorch inference can run around 20 fps on my Macbook pro, and around 203 fps on Nvidia GeForce 1080Ti.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and that 1280x720 resolution is possible.

**Note**: Currently the intelligent cropping algorithm doesn't work.

#### movenet.ipynb

The notebook is borrowed from the official movenet tutorial. You can go through it for a better understanding of model.

### Movenet deployment

Google releases an Android [demo](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android) for Tensorflow Lite Pose estimation application. Movenet is included in this demo. In order to compare the speed of my own implementation with the original one during the inference phase, I tried to convert the PyTorch Movenet model to Tensorflow Lite model and embed it into the Android demo. However, there're many issues during the conversion. 

**Update**: The great work [TinyNeuralNetwork](https://github.com/alibaba/TinyNeuralNetwork) solves all problems! Thanks for their help!!! Now the PyTorch model can be directly converted into TFLite model. Check [export.py](export.py).

To deploy the model, what you need to do:
1. Use [export.py](export.py) to convert the PyTorch model to TFlite model. The model will be stored in the directory `_models`.
2. Move the models to the directory `android/app/src/main/assets`.
3. Compile the android project, then you can see how the MoveNet performs on your phone.


### 3D Pose Estimation
This repo also exploits 2d-to-3d pose estimation, of which the inputs are obtained from MoveNet.

The current 2D-to-3D model is from [PoseAug](https://github.com/jfzhang95/PoseAug) and the model structure I used is [VideoPose](https://github.com/facebookresearch/VideoPose3D). 

Run the webcom [demo](webcam_demo_3d.py) to see the visualizaton of 3D-pose.
