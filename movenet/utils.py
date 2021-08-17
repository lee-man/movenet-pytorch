import cv2
import numpy as np

import movenet.constants as constants


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, size=192):
    # stat= {'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278]}
    input_img = cv2.resize(source_img, (size, size), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # input_img = input_img * (2.0 / 255.0) - 1.0
    # input_img = (input_img - stat['mean']) / stat['std']
    # input_img = input_img * 0.007843137718737125 - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, size, size)
    # print(input_img.shape)

    return input_img, source_img


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, size=192):
    img = cv2.imread(path)
    # print(img.shape)
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
    for left, right in constants.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left, :] * 513, keypoint_coords[right, :] * 513]).astype(np.int32),
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


# def draw_skel_and_kp(
#         img, instance_scores, keypoint_scores, keypoint_coords,
#         min_pose_score=0.5, min_part_score=0.5):

#     out_img = img
#     adjacent_keypoints = []
#     cv_keypoints = []
#     for ii, score in enumerate(instance_scores):
#         if score < min_pose_score:
#             continue

#         new_keypoints = get_adjacent_keypoints(
#             keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
#         adjacent_keypoints.extend(new_keypoints)

#         for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
#             if ks < min_part_score:
#                 continue
#             cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

#     if cv_keypoints:
#         out_img = cv2.drawKeypoints(
#             out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
#             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
#     return out_img


def draw_skel_and_kp(
        img, kpt_with_conf, min_part_score=0.2):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []

    keypoint_scores = kpt_with_conf[:, 2]
    keypoint_coords = kpt_with_conf[:, :2]
    new_keypoints = get_adjacent_keypoints(
        keypoint_scores, keypoint_coords, min_part_score)
    adjacent_keypoints.extend(new_keypoints)
    for ks, kc in zip(keypoint_scores, keypoint_coords):
        if ks < min_part_score:
            continue
        cv_keypoints.append(cv2.KeyPoint(kc[0], kc[1], 10. * ks))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img