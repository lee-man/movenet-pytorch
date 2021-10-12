import cv2
import time
import argparse
import os
import torch

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
    model = load_model(args.model)
    model.eval()


    ####### step 1: Pytorch to ONNX #######
    print('==> Start to convert the model from Pytorch to ONNX...')
    # prepare the dummy input
    sample_input = torch.rand((1, 3, 192, 192))    
    # export to ONNX format
    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,                    # Input tensor
        onnx_model_path,        # Output file (eg. 'output_model.onnx')
        opset_version=12,       # Operator support version
        input_names=['input'],   # Input tensor name (arbitary)
        output_names=['output'] # Output tensor name (arbitary)
    )
    print('==> finshed converting the model from Pytorch to ONNX...')
    print('The model is saved in %s' % onnx_model_path )
    exit()

    ####### step 2: ONNX to TF #######
    print('==> Start to convert the model from ONNX to TF...')
    # load the ONNX model
    import onnx
    onnx_model = onnx.load(onnx_model_path)

    # convert with onnx-tf:
    from onnx_tf.backend import prepare
    tf_rep = prepare(onnx_model)

    # export TF model
    tf_rep.export_graph(tf_model_path)
    print('==> finshed converting the model from ONNX to TF...')
    print('The model is saved in %s' % tf_model_path, '.pb')

    ####### step 3: TF to TFLite #######
    print('==> Start to convert the model from TF to TFLite...')
    import tensorflow as tf
    # convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print('==> finshed converting the model from TF to TFLite...')
    print('The model is saved in %s' % tflite_model_path)

if __name__ == "__main__":
    main()