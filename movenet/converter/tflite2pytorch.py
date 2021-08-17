import json
import struct
import cv2
import numpy as np
import os
import tempfile
import torch


BASE_DIR = os.path.join('./movenet', 'weights')


def to_torch_name(name):
    # tf_name = tf_name.lower()
    # tf_split = tf_name.split('/')
    # tf_layer_split = tf_split[1].split('_')
    # tf_variable_type = tf_split[2]
    # if tf_variable_type == 'weights' or tf_variable_type == 'depthwise_weights':
    #     variable_postfix = '.weight'
    # elif tf_variable_type == 'biases':
    #     variable_postfix = '.bias'
    # else:
    #     variable_postfix = ''

    # if tf_layer_split[0] == 'conv2d':
    #     torch_name = 'features.conv' + tf_layer_split[1]
    #     if len(tf_layer_split) > 2:
    #         torch_name += '.' + tf_layer_split[2]
    #     else:
    #         torch_name += '.conv'
    #     torch_name += variable_postfix
    # else:
    #     if tf_layer_split[0] in ['offset', 'displacement', 'heatmap'] and tf_layer_split[-1] == '2':
    #         torch_name = '_'.join(tf_layer_split[:-1])
    #         torch_name += variable_postfix
    #     else:
    #         torch_name = ''
    
    name = name.replace('_', '.')
    if 'hm.hp' in name:
        name = name.replace('hm.hp', 'hm_hp')
    if 'hp.offset' in name:
        name = name.replace('hp.offset', 'hp_offset')
    if 'inner.blocks' in name:
        name = name.replace('inner.blocks', 'inner_blocks')
    if 'layer.blocks' in name:
        name = name.replace('layer.blocks', 'layer_blocks')
    
    return name[:-4]


def load_variables(chkpoint, base_dir=BASE_DIR):
    files = [f for f in os.listdir(base_dir)]

    state_dict = {}
    for filename in files:
        if filename[-4:] != '.npy':
            continue
        torch_name = to_torch_name(filename)
        if not torch_name:
            continue
        d = np.load(os.path.join(base_dir, filename))
        shape = d.shape
        if len(shape) == 4 and 'backbone.body.0.1.weight' not in torch_name:
            # tpt = (2, 3, 0, 1) if 'depthwise' in filename else (3, 2, 0, 1)
            tpt = (3, 0, 1, 2) if (shape[1] == 3 and shape[2] == 3) else (0, 3 ,1, 2)
            d = d.transpose(tpt)
        elif 'backbone.body.0.1.weight' in torch_name:
            tpt = (0, 3 ,1, 2)
            d = d.transpose(tpt)
        state_dict[torch_name] = torch.from_numpy(d)

    return state_dict


def _read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img * (2.0 / 255.0) - 1.0
    img = img.transpose((2, 0, 1))
    return img


def convert(model_id, model_dir, image_size=192, check=True):
    checkpoint_name = model_id
    width = image_size
    height = image_size

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    state_dict = load_variables(checkpoint_name)
    # m = MobileNetV1(model_id, output_stride=output_stride)
    # m.load_state_dict(state_dict)
    checkpoint_path = os.path.join(model_dir, checkpoint_name) + '.pth'
    torch.save(state_dict, checkpoint_path)

    # if check and os.path.exists("./images/tennis_in_crowd.jpg"):
    #     # Result
    #     input_image = _read_imgfile("./images/tennis_in_crowd.jpg", width, height)
    #     input_image = np.array(input_image, dtype=np.float32)
    #     input_image = input_image.reshape(1, 3, height, width)
    #     input_image = torch.Tensor(input_image)

    #     heatmaps_result, offset_result, displacement_fwd_result, displacement_bwd_result = m(input_image)

    #     print("Heatmaps")
    #     print(heatmaps_result.shape)
    #     print(heatmaps_result[:, 0:1, 0:1])
    #     print(torch.mean(heatmaps_result))
