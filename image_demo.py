import cv2
import time
import argparse
import os
import torch

import movenet
from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp
from movenet.decode_single import decode_single_pose



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet")
# parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--notxt', action='store_true')
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
            # 'hm': 1, 'hps': 34, 'hm_hp': 17, 'hp_offset': 34
            # kpt_heatmap, center, kpt_regress, kpt_offset = output['hm_hp'], output['hm'], output['hps'], output['hp_offset']

            # pose_scores, keypoint_scores, keypoint_coords = decode_single_pose(
            #     kpt_heatmap,
            #     center,
            #     kpt_regress,
            #     kpt_offset,
            #     max_pose_detections=10,
            #     min_pose_score=0.25)

        # keypoint_coords *= output_scale

        if args.output_dir:
            draw_image = draw_skel_and_kp(
                draw_image, kpt_with_conf, min_part_score=0.1)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

        # pose_scores = kpt_with_conf[:, 2]
        # if not args.notxt:
        #     print()
        #     print("Results for image: %s" % f)
        #     for pi in range(len(pose_scores)):
        #         if pose_scores[pi] == 0.:
        #             break
        #         print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
        #         for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
        #             print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

    print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
