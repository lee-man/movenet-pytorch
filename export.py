import cv2
import time
import argparse
import os
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.quantization import fuse_modules

from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet")
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--conf_thres', type=float, default=0.3)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    # define some paths
    onnx_model_path = '_models/movenet.onnx'
    tf_model_path = '_models/movenet'
    tflite_model_path = '_models/movenet.tflite'

    # load the PyTorch model
    print('==> Load the Pytorch model...')
    model = load_model(args.model)
    model.eval()

    # fuse common operations. Current model definition doesn't utilize
    # print('==> Fuse the common operations: conv + rulu; conv + relu6')
    # modules_to_fuse = []
    # modules_to_fuse.append([''])

    # trace the input to the model
    print('==> jit tracing...')
    example = torch.rand(1, 3, 192, 192)
    traced_script_module = torch.jit.trace(model, example)

    # optimize the model for the mobile
    print('==> optimze for the mobile application')
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)

    # export the model
    print('==> export the model')
    traced_script_module_optimized._save_for_lite_interpreter("android/app/src/main/assets/movenet_pytorch.ptl")




if __name__ == "__main__":
    main()
