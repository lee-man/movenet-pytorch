import cv2
import time
import argparse
import os
import torch

from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet_lightning", choices=["movenet_lightning", "movenet_thunder"])
# parser.add_argument('--size', type=int, default=192)
parser.add_argument('--conf_thres', type=float, default=0.3)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
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

            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()

        if args.output_dir:
            draw_image = draw_skel_and_kp(
                draw_image, kpt_with_conf, conf_thres=args.conf_thres)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)


    print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
