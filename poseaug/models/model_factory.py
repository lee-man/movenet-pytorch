import torch
import os


from poseaug.models.VideoPose3D import get_pose_net

MODEL_DIR = './_models/videopose'
DEBUG_OUTPUT = False



def load_model(model_id="videopose", model_dir=MODEL_DIR):
    assert model_id == 'videopose', 'The model name should be videopose'
    model_path = os.path.join(model_dir, model_id + '.pth')

    model = get_pose_net()

    state_dict = torch.load(model_path)

    
    model.load_state_dict(state_dict)

    return model

