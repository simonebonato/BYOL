import sys

import torch
from karies.config import ModelTypes
from karies.models import U_Net_Model

sys.path.append("/cluster/group/karies_2022/Simone/karies/karies-models/")
from configs.UNet.baseline_unet_config import model_config as config_unet

# ##### HYPERPARAMS #####
batch_size = 4

ENCODER_WEIGHTS_FOLDER = "/cluster/group/karies_2022/Simone/karies/karies-models/BYOL/unet_pixpro_weights/"
ENCODER_WEIGHTS_FILE = "encoder_batch_size_50_crop_size_56_epoch_400.pth"

config_unet["name"] = "PixPro_" + ENCODER_WEIGHTS_FILE
config_unet["fix_random_seed"] = 42
config_unet["unet_type"] = ModelTypes.U_Net
config_unet["batch_size"] = batch_size

# ##### TRAINING #####
unet = U_Net_Model(config_unet, load=False)
ENCODER_WEIGHTS = ENCODER_WEIGHTS_FOLDER + ENCODER_WEIGHTS_FILE
unet.model.encoder.load_state_dict(torch.load(ENCODER_WEIGHTS))

unet.train()
