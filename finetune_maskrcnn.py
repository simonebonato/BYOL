import torch
from karies.config import MaskRCNNConfig, ModelConfig, ModelTypes, Task
from karies.models import MaskRCNN

base_config: ModelConfig = {
    "task": Task.training,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_workers": 0,
    "device": "cuda",
    "num_epochs": 100,
    "image_shape": [768, 1024],
    "dataset": "dataset_4k",
    "labels_json": "labels_caries.json",
    "histogram_eq": False,
    "visualization_frequency": 1,
    "augmentations": [],
    "fix_random_seed": 42,
    "path_model": "",
    "load_model_name": "",
}

config: MaskRCNNConfig = {
    **base_config,
    "model_type": ModelTypes.MaskRCNN,
    "classes": 5,
    "iou_threshold": 0.1,
    "confidence_threshold": 0.1,
    "model_args": {},
    "loss_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
}


# ######### PRETRAINED WEIGHTS LOAD PATH #########
pixpro_weights = "/cluster/group/karies_2022/Simone/karies/karies-models/AAA_BYOL_test/BYOL/maskrcnn_pixpro_weights/pretrained_weights_0_epochs.pth"

# ######### OTHER TRAINING PARAMS #########
freeze_backbone = False
config["batch_size"] = 4
config["name"] = "PixPro-1-pretrain"

config["name"] += "-FROZEN" if freeze_backbone else ""

maskrcnn = MaskRCNN(config, load=False)
maskrcnn.model.load_state_dict(torch.load(pixpro_weights), strict=False)

if freeze_backbone:
    for layer in [maskrcnn.model.transform, maskrcnn.model.backbone]:
        for weight in layer.parameters():
            weight.requires_grad = False

maskrcnn.train()
