import torch
from poseaug.models.model_factory import load_model as load_model_pose
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


print(">>> creating and load model")
# model = LinearModel()
model = load_model_pose()

print(">>> convert torch model to onnx model")
dummy_input = torch.randn(1, 16, 2)
torch.onnx.export(model, dummy_input, "_models/videopose/videopose.onnx")

print(">>> convert onnx model to tensorflow model")
model = onnx.load("_models/videopose/videopose.onnx")
model_tf = prepare(model)
model_tf.export_graph("_models/videopose/videopose.pb")

print(">>> convert the saved model to TF-Lite model")

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("_models/videopose/videopose.pb") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('_models/videopose/videopose.tflite', 'wb') as f:
  f.write(tflite_model)