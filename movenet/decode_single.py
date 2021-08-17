from movenet.decode import *
from movenet.constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F



def decode_single_pose(kpt_heatmap, offsets, displacements_fwd, displacements_bwd):

    # perform part scoring step on GPU as it's expensive
    # TODO determine how much more of this would be worth performing on the GPU
    part_scores, part_idx = build_part_with_score_torch(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    part_scores = part_scores.cpu().numpy()
    part_idx = part_idx.cpu().numpy()

    scores = scores.cpu().numpy()
    height = scores.shape[1]
    width = scores.shape[2]
    # change dimensions from (x, h, w) to (x//2, h, w, 2) to allow return of complete coord array
    offsets = offsets.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_fwd = displacements_fwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_bwd = displacements_bwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))

    squared_nms_radius = nms_radius ** 2
    pose_count = 0
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

    for root_score, (root_id, root_coord_y, root_coord_x) in zip(part_scores, part_idx):
        root_coord = np.array([root_coord_y, root_coord_x])
        root_image_coords = root_coord * output_stride + offsets[root_id, root_coord_y, root_coord_x]

        if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score, root_id, root_image_coords,
            scores, offsets, output_stride,
            displacements_fwd, displacements_bwd)

        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :], squared_nms_radius, keypoint_scores, keypoint_coords)

        # NOTE this isn't in the original implementation, but it appears that by initially ordering by
        # part scores, and having a max # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords