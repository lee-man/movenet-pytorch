import cv2
import numpy as np

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

from movenet.constants import *


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, size=192):
    input_img = cv2.resize(source_img, (size, size),
                           interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, size, size)

    return input_img, source_img


def read_cap(cap, size=192):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, size)


def read_imgfile(path, size=192):
    img = cv2.imread(path)
    return _process_input(img, size)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results

def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, kpt_with_conf, conf_thres=0.1):

    out_img = img
    height, width, _ = img.shape
    adjacent_keypoints = []
    cv_keypoints = []

    keypoint_scores = kpt_with_conf[:, 2]
    keypoint_coords = kpt_with_conf[:, :2]
    keypoint_coords[:, 0] = keypoint_coords[:, 0] * height
    keypoint_coords[:, 1] = keypoint_coords[:, 1] * width

    new_keypoints = get_adjacent_keypoints(
        keypoint_scores, keypoint_coords, conf_thres)
    adjacent_keypoints.extend(new_keypoints)
    for ks, kc in zip(keypoint_scores, keypoint_coords):
        if ks < conf_thres:
            continue
        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 5))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img
