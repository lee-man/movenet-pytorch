import torch
import cv2
import time
import argparse

from movenet.models.model_factory import load_model
from movenet.utils import read_cap, draw_skel_and_kp
# cropping related functions
from movenet.utils import init_crop_region, determine_crop_region

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet")
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--conf_thres', type=float, default=0.3)
parser.add_argument('--cropping', action='store_false')
args = parser.parse_args()


def main():
    model = load_model(args.model)
    # model = model.cuda()

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0

    if args.cropping:
        crop_region = init_crop_region(args.cam_height, args.cam_width)
    while True:
        if not args.cropping:
            input_image, display_image = read_cap(
            cap, size=args.size)

            with torch.no_grad():
                input_image = torch.Tensor(input_image) #.cuda()

                kpt_with_conf = model(input_image)[0, 0, :, :]
                kpt_with_conf = kpt_with_conf.numpy()
        else:
            input_image, display_image = read_cap(
            cap, size=args.size, crop_region=crop_region)
            with torch.no_grad():
                input_image = torch.Tensor(input_image) #.cuda()

                kpt_with_conf = model(input_image)[0, 0, :, :]
                for idx in range(17):
                    kpt_with_conf[idx, 0] = (
                        crop_region['y_min'] +
                        crop_region['height'] * args.cam_height *
                        kpt_with_conf[idx, 0]) / args.cam_height
                    kpt_with_conf[idx, 1] = (
                        crop_region['x_min'] +
                        crop_region['width'] * args.cam_width *
                        kpt_with_conf[idx, 1]) / args.cam_width
                kpt_with_conf = kpt_with_conf.numpy()
                crop_region = determine_crop_region(
                    kpt_with_conf, args.cam_height, args.cam_width)

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