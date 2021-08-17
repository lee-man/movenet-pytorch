import cv2
import time
import argparse
import os
import torch

import movenet
from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_prediction_on_image
from movenet.decode_single import decode_single_pose



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet")
# parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--conf_thres', type=float, default=0.1)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    model = load_model(args.model)
    # model = model.cuda()

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg', 'jpeg'))]

    start = time.time()
    for f in filenames:
        input_image, draw_image = read_imgfile(
            f, args.size)

        with torch.no_grad():
            input_image = torch.Tensor(input_image) # .cuda()

            kpt_with_conf = model.decode(input_image)
            kpt_with_conf = kpt_with_conf.numpy()

        if args.output_dir:
            draw_image = draw_prediction_on_image(
                draw_image, kpt_with_conf, conf_thres=args.conf_thres)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)


    print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
