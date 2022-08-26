# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import os.path as osp
import sys
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms

# from PIL import Image
from alfred import device

# Type agnostic hand detector
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.utils.visualizer import Visualizer

# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
# from detectron2.modeling import GeneralizedRCNNWithTTA
# from detectron2.data.datasets import register_coco_instances

# Type-aware hand (hand-object) hand detector
hand_object_detector_path = "./detectors/yolo_v3"
sys.path.append(hand_object_detector_path)
from yolov3 import Yolov3, Yolov3Tiny
from predict import process_data 
from utils.utils import non_max_suppression, scale_coords


class YOLOv3Detector:
    def __init__(self) -> None:
        self.cfg = 'yolo3-tiny'
        self.img_size = 416

        torch.set_grad_enabled(False)
        self.__load_hand_detector()

    def __load_hand_detector(self):
        weights = "detectors/yolo_v3/hand-tiny_512-2021-02-19.pt"
        if "-tiny" in self.cfg:
            a_scalse = 416.0 / self.img_size
            anchors = [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
            anchors_new = [
                (int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse))
                for j in range(len(anchors))
            ]
            self.hand_detector = Yolov3Tiny(1, anchors=anchors_new)
        else:
            a_scalse = 416.0 / self.img_size
            anchors = [
                (10, 13),
                (16, 30),
                (33, 23),
                (30, 61),
                (62, 45),
                (59, 119),
                (116, 90),
                (156, 198),
                (373, 326),
            ]
            anchors_new = [
                (int(anchors[j][0] / a_scalse), int(anchors[j][1] / a_scalse))
                for j in range(len(anchors))
            ]
            self.hand_detector = Yolov3(1, anchors=anchors_new)

        use_cuda = torch.cuda.is_available()
        # Load weights
        if os.access(weights, os.F_OK):  # 判断模型文件是否存在
            self.hand_detector.load_state_dict(torch.load(weights, map_location=device)["model"])
        else:
            print("error model not exists")
            return False
        self.hand_detector.to(device).eval()  # 模型模式设置为 eval


    def __get_raw_hand_bbox(self, img):
        img_in = process_data(img, self.img_size)
        img_in = torch.from_numpy(img_in).unsqueeze(0).to(device)
        pred, _ = self.hand_detector(img_in)#图片检测
        detections = non_max_suppression(pred, 0.5, 0.65)[0] # nms
        if detections is not None:
            print(detections.shape)
            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], img.shape).round()
            detections = detections[:, :4].cpu().numpy()
        else:
            detections = np.array([])
        return detections
            

    def detect_hand_bbox(self, img):
        """
        output:
            body_bbox: [min_x, min_y, width, height]
            hand_bbox: [x0, y0, x1, y1]
        Note:
            len(body_bbox) == len(hand_bbox), where hand_bbox can be None if not valid
        """
        # get body pose
        # body_pose_list, body_bbox_list = self.detect_body_pose(img)
        # assert len(body_pose_list) == 1, "Current version only supports one person"

        # get raw hand bboxes
        raw_hand_bboxes = self.__get_raw_hand_bbox(img)
        num_bbox = raw_hand_bboxes.shape[0]
        hand_bbox_list = [None]

        if num_bbox > 0:
            dist_left_arm = np.ones((num_bbox,)) * float("inf")
            dist_right_arm = np.ones((num_bbox,)) * float("inf")
            hand_bboxes = dict(left_hand=None, right_hand=None)
            # assign bboxes
            # hand_bboxes = dict()
            left_id = 0
            right_id = 1 if num_bbox > 1 else 0

            hand_bboxes["left_hand"] = raw_hand_bboxes[left_id].copy()
            hand_bboxes["right_hand"] = raw_hand_bboxes[right_id].copy()

            hand_bbox_list[0] = hand_bboxes
        return [], [], hand_bbox_list, raw_hand_bboxes


class Third_View_Detector():
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    with a type-agnostic hand detector (https://github.com/ddshan/hand_detector.d2)
    """

    def __init__(self):
        super(Third_View_Detector, self).__init__()
        print("Loading Third View Hand Detector")
        self.__load_hand_detector()

    def __load_hand_detector(self):
        # load cfg and model
        cfg = get_cfg()
        cfg.merge_from_file(
            "detectors/hand_only_detector/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml"
        )
        cfg.MODEL.WEIGHTS = "extra_data/hand_module/hand_detector/model_0529999.pth"  # add model weight here
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            0.3  # 0.3 , use low thresh to increase recall
        )
        self.hand_detector = DefaultPredictor(cfg)

    def __get_raw_hand_bbox(self, img):
        bbox_tensor = self.hand_detector(img)["instances"].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        return bboxes

    def detect_hand_bbox(self, img):
        """
        output:
            body_bbox: [min_x, min_y, width, height]
            hand_bbox: [x0, y0, x1, y1]
        Note:
            len(body_bbox) == len(hand_bbox), where hand_bbox can be None if not valid
        """
        # get body pose
        body_pose_list, body_bbox_list = self.detect_body_pose(img)
        # assert len(body_pose_list) == 1, "Current version only supports one person"

        # get raw hand bboxes
        raw_hand_bboxes = self.__get_raw_hand_bbox(img)
        hand_bbox_list = [
            None,
        ] * len(body_pose_list)
        num_bbox = raw_hand_bboxes.shape[0]

        if num_bbox > 0:
            for idx, body_pose in enumerate(body_pose_list):
                # By default, we use distance to ankle to distinguish left/right,
                # if ankle is unavailable, use elbow, then use shoulder.
                # The joints used by two arms should exactly the same)
                dist_left_arm = np.ones((num_bbox,)) * float("inf")
                dist_right_arm = np.ones((num_bbox,)) * float("inf")
                hand_bboxes = dict(left_hand=None, right_hand=None)
                # left arm
                if body_pose[7][0] > 0 and body_pose[6][0] > 0:
                    # distance between elbow and ankle
                    dist_wrist_elbow = np.linalg.norm(body_pose[7] - body_pose[6])
                    for i in range(num_bbox):
                        bbox = raw_hand_bboxes[i]
                        c_x = (bbox[0] + bbox[2]) / 2
                        c_y = (bbox[1] + bbox[3]) / 2
                        center = np.array([c_x, c_y])
                        dist_bbox_ankle = np.linalg.norm(center - body_pose[7])
                        if dist_bbox_ankle < dist_wrist_elbow * 1.5:
                            dist_left_arm[i] = np.linalg.norm(center - body_pose[7])
                # right arm
                if body_pose[4][0] > 0 and body_pose[3][0] > 0:
                    # distance between elbow and ankle
                    dist_wrist_elbow = np.linalg.norm(body_pose[3] - body_pose[4])
                    for i in range(num_bbox):
                        bbox = raw_hand_bboxes[i]
                        c_x = (bbox[0] + bbox[2]) / 2
                        c_y = (bbox[1] + bbox[3]) / 2
                        center = np.array([c_x, c_y])
                        dist_bbox_ankle = np.linalg.norm(center - body_pose[4])
                        if dist_bbox_ankle < dist_wrist_elbow * 1.5:
                            dist_right_arm[i] = np.linalg.norm(center - body_pose[4])

                # assign bboxes
                # hand_bboxes = dict()
                left_id = np.argmin(dist_left_arm)
                right_id = np.argmin(dist_right_arm)

                if dist_left_arm[left_id] < float("inf"):
                    hand_bboxes["left_hand"] = raw_hand_bboxes[left_id].copy()
                if dist_right_arm[right_id] < float("inf"):
                    hand_bboxes["right_hand"] = raw_hand_bboxes[right_id].copy()

                hand_bbox_list[idx] = hand_bboxes

        assert len(body_bbox_list) == len(hand_bbox_list)
        return body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes


class HandBboxDetector(object):
    def __init__(self, view_type, device):
        """
        args:
            view_type: third_view or ego_centric.
        """
        self.view_type = view_type
        # self.model = Third_View_Detector()
        self.model = YOLOv3Detector()

    def detect_body_bbox(self, img_bgr):
        return self.model.detect_body_pose(img_bgr)

    def detect_hand_bbox(self, img_bgr):
        """
        args:
            img_bgr: Raw image with BGR order (cv2 default). Currently assumes BGR
        output:
            body_pose_list: body poses
            bbox_bbox_list: list of bboxes. Each bbox has XHWH form (min_x, min_y, width, height)
            hand_bbox_list: each element is
            dict(
                left_hand = None / [min_x, min_y, width, height]
                right_hand = None / [min_x, min_y, width, height]
            )
            raw_hand_bboxes: list of raw hand detection, each element is [min_x, min_y, width, height]
        """
        output = self.model.detect_hand_bbox(img_bgr)
        body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = output

        # convert raw_hand_bboxes from (x0, y0, x1, y1) to (x0, y0, w, h)
        if raw_hand_bboxes is not None:
            for i in range(raw_hand_bboxes.shape[0]):
                bbox = raw_hand_bboxes[i]
                x0, y0, x1, y1 = bbox
                raw_hand_bboxes[i] = np.array([x0, y0, x1 - x0, y1 - y0])

        # convert hand_bbox_list from (x0, y0, x1, y1) to (x0, y0, w, h)
        for hand_bbox in hand_bbox_list:
            if hand_bbox is not None:
                for hand_type in hand_bbox:
                    bbox = hand_bbox[hand_type]
                    if bbox is not None:
                        x0, y0, x1, y1 = bbox
                        hand_bbox[hand_type] = np.array([x0, y0, x1 - x0, y1 - y0])

        return body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes
