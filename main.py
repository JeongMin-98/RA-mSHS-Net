# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------
#
# from model.model import check_model_build, run_visualize_feature_map_func, DeepNetwork
import argparse
import os
import pprint

from torchinfo import summary

# The import statement below will be refactored soon.
import _init_path
from model import get_pose_net
from config import cfg
from config import update_config
from utils.tools import check_device
from utils.tools import create_logger


def parse_args():
    desc = "Pytorch implementation of DeepNetwork"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--cfg',
                        default='experiments/pose_resnet.yaml',
                        help='experiment configure file name',
                        required=False,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def run_fn(config):
    device = check_device()

    # logger, final_output_dir, tb_log_dir = create_logger(
    #     config, args.cfg, 'train'
    # )
    # init model
    model = get_pose_net(config, is_train=True)


"""main"""


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(cfg))

    # # cudnn related setting
    # cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = get_pose_net(cfg, is_train=True)

    # logger
    # create logger

    # run
    # run_fn(cfg)
    # check_model_build(args=args)
    # run_visualize_feature_map_func(args)


if __name__ == '__main__':
    main()
