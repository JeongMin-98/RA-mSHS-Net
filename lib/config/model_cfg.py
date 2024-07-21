# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# Your model related params
# examples
# FCN = CN(new_allowed=True)
# FCN.INPUT_CHANNELS = 784
# FCN.HIDDEN_CHANNELS = [128, 64, 10]
# FCN.HIDDEN_ACTIVATION = 'ReLU'
# FCN.HIDDEN_DROPOUT = 0.25
# FCN.OUTPUT_CHANNELS = 2
# FCN.OUTPUT_ACTIVATION = 'logSoftMax'

POSE_RESNET = CN(new_allowed=True)
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
POSE_RESNET.SIGMA = 2

MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
}
