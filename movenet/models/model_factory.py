import torch
import os


from movenet.models.movenet import get_pose_net

MODEL_DIR = './_models'
DEBUG_OUTPUT = False

heads = {'hm': 1, 'hps': 34, 'hm_hp': 17, 'hp_offset': 34}


def load_model(model_id, output_stride=4, ft_size=48, model_dir=MODEL_DIR):
    assert model_id in ['movenet_lightning', 'movenet_thunder'], 'The model name should be movenet_lightning or movenet_thudner'
    assert output_stride == 4, 'The current model only support output stride being 4.'
    model_path = os.path.join(model_dir, model_id + '.pth')
    if not os.path.exists(model_path):
        print('Cannot find models file %s, converting from tflite weights...' % model_path)
        from movenet.converter.tflite2pytorch import convert
        convert(model_id, model_dir, check=False)
        assert os.path.exists(model_path)

    model = get_pose_net(0, heads, model_type=model_id, ft_size=ft_size)
    model_state_dict = model.state_dict()

    state_dict = torch.load(model_path)

    # Borrowed from centernet
    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape {}, '\
                    'loaded shape{}. {}'.format(
                k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model
