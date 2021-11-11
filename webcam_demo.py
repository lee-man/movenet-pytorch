import torch
import cv2
import time
import argparse

from movenet.models.model_factory import load_model
from movenet.utils import read_cap, draw_skel_and_kp
# cropping related functions
# from movenet.utils import init_crop_region, determine_crop_region

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet_lightning", choices=["movenet_lightning", "movenet_thunder"])
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--conf_thres', type=float, default=0.3)
args = parser.parse_args()

if args.model == "movenet_lightning":
    args.size = 192
    args.ft_size = 48
else:
    args.size = 256
    args.ft_size = 64

def main():
    model = load_model(args.model, ft_size=args.ft_size)
    # model = model.cuda()

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0

    while True:
        input_image, display_image = read_cap(
        cap, size=args.size)
        with torch.no_grad():
            input_image = torch.Tensor(input_image) #.cuda()
            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()

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