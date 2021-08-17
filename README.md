# Movenet Pytorch

This repository contains a PyTorch implementation of [Movenet](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) from Google. The inference correctness is the point this repo focuses on. If you are interested in Movenet training, you can refer to my another repo [lee-man/movenet](https://github.com/lee-man/movenet).

### Credits

The original model, weights, etc. was created by Google and can be found at [tf hub](https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3).

I will only released the model definition and pose decoding part of Movenet. No weights will be provided.

Portions of the code in this repo are borrowed from the following repos:
1. [Centernet](https://github.com/xingyizhou/CenterNet) for prediction heads.
2. [Pytorch Vision](https://github.com/pytorch/vision) for MobileNet and Feature Pyramid Network.
3. [posenet-pytorch](https://github.com/rwightman/posenet-pytorch) for the other utility functions.
4. [TF official tutorial for Movenet](https://www.tensorflow.org/hub/tutorials/movenet) for image drawing.


### Notice
In order to get in touch with the internal computational flow of Movenet, I use [netron](https://github.com/lutzroeder/netron) to visualize Movenet and my own implementation. You can use netron to compare the one with another. The model definitions in onnx format or tflite format are maintained in `_models` directory. There are a few things to pay attention to when I try to convert the movenet TFLite model to PyTorch model:

* Extract the weights:
  * Currently I use the most clumsy way to extract the weights from TFLite model: open it with netron, select the layers/ops I want to deal with, export the weights, and rename the numpy file using PyTorch convention.
* Batchnorm:
  * As the TFLite fuses the convolutional layers and batchnorm layers together, the BatchNorm layers are removed from PyTorch MobileNet v2 implementation.
* Padding:
  * The ways how Tensorflow and PyTorch do the padding are slightly different. In Tensorflow and original Movenet, the padding mode is specified as `same`, and it will do the asymmetry padding with the setting: `stride=2, kernel_size=3`. But in PyTorch, the padding will be symmetry. So I add `nn.ZeroPad2d` layers before Conv2d layer with `stride=2, kernel_size=3`.
* Feature Pyramid Network:
  * The structure of FPN used in Movenet is different from the official implementation in Pytorch Vision. In Movenet, the size of feature map for top-donwn structure of FPN is incrementally increased.
  * The upsampling mechanism used in Movenet is `bilinear` interpolatation with `align_corners=False, half_pixel_centers=True`. It's fine to use this setting under PyTorch inference. But it seems that ONNX doesn't support this kind of operation yet, so I change it to `nearest` mode when I convert the model to ONNX format.

### Install

A suitable Python 3.x environment with a recent version of PyTorch is required. Development and testing was done with Python 3.7.1 and PyTorch 1.0 w/ CUDA10 from Conda.

If you want to use the webcam demo, a pip version of opencv (`pip install python-opencv=3.4.5.20`) is required instead of the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. The python bindings for OpenCV 4.0 currently have a broken impl of drawKeypoints so please force install a 3.4.x version.

A fresh conda Python 3.6/3.7 environment with the following installs should suffice: 
```
conda install -c pytorch pytorch cudatoolkit
pip install requests opencv-python==3.4.5.20
```

### Usage

There are three demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

For all demos, the model can be specified with the '--model` argument by using its integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and that 1280x720 resolution is possible.

### Credits

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port and my work is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at https://github.com/infocom-tpo/PoseNet-CoreML

### TODO (someday, maybe)
* More stringent verification of correctness against the original implementation
* Performance improvements (especially edge loops in 'decode.py')
* OpenGL rendering/drawing
* Comment interfaces, tensor dimensions, etc
* Implement batch inference for image_demo
* Create a training routine and add models with more advanced CNN backbones

