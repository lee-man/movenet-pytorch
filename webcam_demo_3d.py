import torch
import cv2
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from movenet.models.model_factory import load_model
from movenet.utils import read_cap, draw_skel_and_kp
# cropping related functions
# from movenet.utils import init_crop_region, determine_crop_region

# videopose
from poseaug.models.model_factory import load_model as load_model_pose
from poseaug.utils import create_2d_data, show3Dpose

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet")
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1000)
parser.add_argument('--cam_height', type=int, default=1000)
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--conf_thres', type=float, default=0.3)
parser.add_argument('--cropping', action='store_false')
args = parser.parse_args()


def main():
    model = load_model(args.model)
    pose_aug = load_model_pose().eval()
    # model = model.cuda()

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    start = time.time()
    frame_count = 0

    while True:
        input_image, display_image = read_cap(
        cap, size=args.size)
        with torch.no_grad():
            input_image = torch.Tensor(input_image) #.cuda()
            kpt_with_conf = model(input_image)[0, 0, :, :]
            inputs_2d = create_2d_data(kpt_with_conf) 
            outputs_3d = pose_aug(inputs_2d)
            outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]
            
        kpt_with_conf = kpt_with_conf.numpy()
        outputs_3d = outputs_3d[0].numpy() 

        show3Dpose(outputs_3d, ax)
        # redraw the canvas
        fig.canvas.draw()
        ax.clear()

        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # display image with opencv or any operation you like
        cv2.imshow("3d", img)
        

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = draw_skel_and_kp(
            display_image, kpt_with_conf, conf_thres=args.conf_thres)

        cv2.imshow('movenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()