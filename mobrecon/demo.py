import sys
import os
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from configs.config import get_cfg

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


if __name__ == "__main__":
    # get config
    args = BaseOptions().parse()
    
    # args = CFGOptions().parse()
    # cfg = setup(args)

    # # dir prepare
    args.work_dir = osp.dirname(osp.realpath(__file__))
    data_fp = osp.join(args.work_dir, "../data", args.dataset)
    args.out_dir = osp.join(args.work_dir, "out", args.dataset, args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, "checkpoints")
    if args.phase in ["eval", "demo"]:
        utils.makedirs(osp.join(args.out_dir, args.phase))
    utils.makedirs(args.out_dir)
    utils.makedirs(args.checkpoints_dir)
    # device set
    torch.set_num_threads(args.n_threads)

    if args.dataset == "Human36M":
        template_fp = osp.join(args.work_dir, "./template/template_body.ply")
        transform_fp = osp.join(args.work_dir, "../template/transform_body.pkl")
    else:
        template_fp = osp.join(args.work_dir, "./template/template.ply")
        transform_fp = osp.join(args.work_dir, "./template/transform.pkl")
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(
        transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation
    )

    if args.model == "mobrecon":
        for i in range(len(up_transform_list)):
            up_transform_list[i] = (
                *up_transform_list[i]._indices(),
                up_transform_list[i]._values(),
            )
        # logger.info(f'spiral_indices_list: {spiral_indices_list}, {up_transform_list}')
        model = MobRecon(args, spiral_indices_list, up_transform_list)
        # model = MobRecon_DS(args)
    else:
        raise Exception("Model {} not support".format(args.model))
    
    # model = MobRecon_DS(cfg)

    # load
    epoch = 0
    if args.resume:
        model_path = args.resume
        print(f"load from model path: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        if checkpoint.get("model_state_dict", None) is not None:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
        epoch = checkpoint.get("epoch", -1) + 1
        cprint("Load checkpoint {}".format(model_path), "yellow")
    model = model.to(device)

    # run
    runner = Runner(args, model, tmp["face"], device)
    runner.set_demo(args)
    runner.demo()
   
