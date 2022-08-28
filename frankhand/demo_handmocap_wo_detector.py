# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from hand_mocap_api import HandMocap
from hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time
from alfred import device

from bodymocap.body_bbox_detector import BodyPoseEstimator
from bodymocap.body_mocap_api import BodyMocap

"""
this will run body mocap
get the SMPL mesh, using SMPL mesh to get
the hands box

without needing a hand box detector

"""


def run_hand_mocap(args, body_bbox_detector, body_mocap, hand_mocap, visualizer):
    # Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)

    assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0

    while True:
        # load data
        load_bbox = False

        if input_type == "image_dir":
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == "bbox_dir":
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]["image_path"]
                hand_bbox_list = input_data[cur_frame]["hand_bbox_list"]
                body_bbox_list = input_data[cur_frame]["body_bbox_list"]
                img_original_bgr = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == "video":
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == "webcam":
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame += 1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break
        print("--------------------------------------")

        # Get handbox_list from body mocap
        body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
            img_original_bgr
        )
        hand_bbox_list = [
            None,
        ] * len(body_bbox_list)

        if len(body_bbox_list) < 1:
            print(f"No body deteced: {image_path}")
            continue

        # Sort the bbox using bbox size
        # (to make the order as consistent as possible without tracking)
        bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
        if args.single_person and len(body_bbox_list) > 0:
            body_bbox_list = [
                body_bbox_list[0],
            ]

        # Body Pose Regression
        pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)

        hand_bbox_list = body_mocap.get_hand_bboxes(
            pred_output_list, img_original_bgr.shape[:2]
        )
        print("hand_bbox_list: ", hand_bbox_list)

        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True
        )
        # assert len(hand_bbox_list) == len(body_bbox_list)
        # assert len(body_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list=pred_mesh_list,
            hand_bbox_list=hand_bbox_list,
        )

        # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save the image (we can make an option here)
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = "hand"
            demo_utils.save_pred_to_pkl(
                args,
                demo_type,
                image_path,
                body_bbox_list,
                hand_bbox_list,
                pred_output_list,
            )

        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ["video", "webcam"]:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    # When everything done, release the capture
    if input_type == "webcam" and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Set Bbox detector
    # bbox_detector = HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device=device)

    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator()

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = (
        args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    )
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    if args.renderer_type in ["pytorch3d", "opendr"]:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, body_bbox_detector, body_mocap, hand_mocap, visualizer)


if __name__ == "__main__":
    main()
