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
  * There are two categories of Movenet: Lightning and Thunder. The current code only supports Lightning ones (version 3). Google haven't released the full-precision version of Lightning version 4. The version 4 model should have improved prediction accuracy as synthetic images with under-represented poses are inlcluded in the training dataset.
* Extract the weights:
  * Currently I use the most clumsy way to extract the weights from TFLite model: open it with netron, select the layers/ops I want to deal with, export the weights, and rename the numpy file using PyTorch convention.
  * Once you extract all the weights, place them under `_models\weights` directory. `movenet.pth` will be generated under `_models` directory when you first run the program. 
* Batchnorm:
  * As the TFLite fuses the convolutional layers and batchnorm layers together, the BatchNorm layers are removed from PyTorch MobileNet v2 implementation.
* Padding:
  * The ways how Tensorflow and PyTorch do the padding are slightly different. In Tensorflow and original Movenet, the padding mode is specified as `same`, and it will do the asymmetry padding with the setting: `stride=2, kernel_size=3`. But in PyTorch, the padding will be symmetry. So I add `nn.ZeroPad2d` layers before Conv2d layer with `stride=2, kernel_size=3`.
* Feature Pyramid Network:
  * The structure of FPN used in Movenet is different from the official implementation in Pytorch Vision. In Movenet, the size of feature map for top-donwn structure of FPN is incrementally increased.
  * The upsampling mechanism used in Movenet is `bilinear` interpolatation with `align_corners=False, half_pixel_centers=True`. It's fine to use this setting under PyTorch inference. But it seems that ONNX doesn't support this kind of operation yet, so I change it to `nearest` mode when I convert the model to ONNX format.

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

The notebook is borrowed from official movenet tutorial. You can go through it for the better understanding of model.

### Movenet deployment

Google releases an Andorid demo for Tensorflow Lite Pose estimtaion application. Movenet is included in this demo. In order to compare the speed of my own implementation with the original one during inference phase, I add some scripts to convert the PyTorch Movenet model to Tensorflow Lite model and embed it into the Android demo.

As there is no direct converter from Pytorch to TFLite, I follow the instructions in [Pytorch-ONNX-TFLite](https://github.com/sithu31296/PyTorch-ONNX-TFLite) to convert the Pytorch model to ONNX, ONNX to Tensorflow SavedModel, Tensorflow SavedModel to TFLite. For the prerequisite libs needed for the conversion, please refer to the original repo.

~~**Important**: Using the current script will result in sub-optimal TFLite models, due to some un-supported ops being replaced by `Flex ops` in TFLite. Also, the fusion of `Conv + ReLU6` and `Conv + ReLU` is not performed.~~

One remaining problem is that going through the above procedure will result a TFLite model requiring inputs with NCHW ordering. The official Movenet TFLite asks for the inputs with shape of NHWC. This is due to different conventions adopted by Tensorflow and Pytorch. Thanks for PINTO0309's work [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow), there's a workaround to transpose the channels. I will check it later and complete the whole workflow from PyTorch model to mobile deployment.

**Update**: I follow the precedure introduced in [PINTO's blog](https://qiita.com/PINTO/items/ed06e03eb5c007c2e102). But I still face some bugs right now. The author help me generate the correct TFLite model (See [Issue](https://github.com/PINTO0309/openvino2tensorflow/issues/66))


I will also try to run [PyTorch Mobiles](https://pytorch.org/mobile/home/) directly and compare its inference speed with TFLite model.

**Update**: I will replace the advanced slicing ops in post-processing with `gather` operation.


### 3D Pose Estimation
This repo also exploits 2d-to-3d pose estimation, of which the inputs are obtained from MoveNet.

The current 2D-to-3D model is from [PoseAug](https://github.com/jfzhang95/PoseAug) and the model structure I used is [VideoPose](https://github.com/facebookresearch/VideoPose3D). 

Run the webcom [demo](webcam_demo_3d.py) to see the visualizaton of 3D-pose.
