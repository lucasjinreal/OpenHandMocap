import pickle
import sys
import os
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from configs.config import get_cfg
from datasets.kinematics import mano_to_mpii
from utils.draw3d import save_a_image_with_mesh_joints
from utils.vis import cnt_area, map2uv, registration
from utils.vis import base_transform

from models.mobrecon_densestack import MobRecon
from utils.read import spiral_tramsform
from utils import utils, writer
from options.base_options import BaseOptions
from torch.utils.data import DataLoader
from core.test_runner import Runner
from termcolor import cprint

from models.mobrecon_ds import MobRecon_DS

from alfred import device, logger
from options.cfg_options import CFGOptions
import numpy as np
import glob
from alfred.utils.file_io import ImageSourceIter
import cv2
from alfred.vis.image.pose_hand import vis_hand_pose, vis_3d_mesh_on_img


def save_a_image_with_mesh_joints(
    image,
    cam_param,
    mesh_xyz,
    face,
    pose_uv,
):
    rend_img_overlay = vis_3d_mesh_on_img(image, cam_param, mesh_xyz, face)
    skeleton_overlay = vis_hand_pose(image, pose_uv)
    # skeleton_3d = draw_3d_skeleton(pose_xyz, image.shape[:2])
    # mesh_3d = vis_3d_mesh_matplot(mesh_xyz, image.shape[:2], face)
    return skeleton_overlay, rend_img_overlay


class DemoVisualizer:
    def __init__(self, cfg, faces) -> None:
        with open(
            os.path.join(os.path.dirname(__file__), "./template/MANO_RIGHT.pkl"), "rb"
        ) as f:
            mano = pickle.load(f, encoding="latin1")
        self.j_regressor = np.zeros([21, 778])
        self.j_regressor[:16] = mano["J_regressor"].toarray()
        for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
            self.j_regressor[k, v] = 1
        self.std = 0.20
        self.size = cfg.DATA.SIZE

        self.K = np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]])
        self.K[0, 0] = self.K[0, 0] / 224 * self.size
        self.K[1, 1] = self.K[1, 1] / 224 * self.size
        self.K[0, 2] = self.size // 2
        self.K[1, 2] = self.size // 2
        self.faces = faces

    def vis(self, pred_mesh):
        vertex = pred_mesh * 0.20
        vertex, align_state = registration(
            vertex,
            uv_point_pred[0],
            self.j_regressor,
            self.K,
            self.size,
            uv_conf=uv_pred_conf[0],
            poly=poly,
        )
        res = save_a_image_with_mesh_joints(
            image[..., ::-1],
            self.K,
            vertex,
            self.faces[0],
            uv_point_pred[0],
        )
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


"""
this is a simplified version of morecon,
running:

python demo_sim.py --config_file configs/mobrecon_ds.yml
"""

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = CFGOptions().parse()
    cfg = setup(args)

    model = MobRecon_DS(cfg)
    logger.info("model loaded.")

    if os.path.exists(cfg.MODEL.RESUME):
        model_path = cfg.MODEL.RESUME
        print(f"load from model path: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        if checkpoint.get("model_state_dict", None) is not None:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
        epoch = checkpoint.get("epoch", -1) + 1
        cprint("Load checkpoint {}".format(model_path), "yellow")
    model = model.to(device)
    model.eval()
    logger.info("model weights loaded.")

    data_f = args.input
    logger.info(f"input: {data_f}")

    iter = ImageSourceIter(data_f)
    visualizer = DemoVisualizer(cfg, model.faces)
    video_save = iter.get_new_video_writter(256, 128, 'a.mp4')
    while iter.ok:
        itm = next(iter)
        if isinstance(itm, str):
            itm = cv2.imread(itm)

        image = itm[..., ::-1]
        image = cv2.resize(image, (cfg.DATA.SIZE, cfg.DATA.SIZE))
        input = (
            torch.from_numpy(base_transform(image, size=cfg.DATA.SIZE))
            .unsqueeze(0)
            .to(device)
        )
        out = model(input)
        mask_pred = out.get("mask_pred")
        if mask_pred is not None:
            mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
            mask_pred = cv2.resize(mask_pred, (input.size(3), input.size(2)))
            try:
                contours, _ = cv2.findContours(
                    mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours.sort(key=cnt_area, reverse=True)
                poly = contours[0].transpose(1, 0, 2).astype(np.int32)
            except:
                poly = None
        else:
            mask_pred = np.zeros([input.size(3), input.size(2)])
            poly = None
        # vertex
        pred = out["verts"][0] if isinstance(out["verts"], list) else out["verts"]
        uv_pred = out["joint_img"]
        if uv_pred.ndim == 4:
            uv_point_pred, uv_pred_conf = map2uv(
                uv_pred.cpu().numpy(), (input.size(2), input.size(3))
            )
        else:
            uv_point_pred, uv_pred_conf = (uv_pred * cfg.DATA.SIZE).cpu().numpy(), [
                None,
            ]
        res = visualizer.vis(pred[0].cpu().numpy())
        cv2.imshow("res_2d", res[0])
        cv2.imshow("res_3dmesh", res[1])
        video_save.write(np.hstack([res[0], res[1][..., :3]]))
        iter.waitKey()
