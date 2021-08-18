import torch
import time
import argparse
import os

from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet")
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--num_images', type=int, default=1000)
args = parser.parse_args()


def main():

    model = load_model(args.model)
    # model = model.cuda()

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg', 'jpeg'))]
    if len(filenames) > args.num_images:
        filenames = filenames[:args.num_images]

    images = {f: read_imgfile(f, args.size)[0] for f in filenames}

    start = time.time()
    for i in range(args.num_images):

        with torch.no_grad():
            input_image = torch.Tensor(images[filenames[i % len(filenames)]]) # .cuda()

            kpt_with_conf = model(input_image)
            kpt_with_conf = kpt_with_conf.numpy()


    print('Average FPS:', args.num_images / (time.time() - start))


if __name__ == "__main__":
    main()
