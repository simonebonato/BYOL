from karies.config import MaskRCNNConfig, ModelConfig, Task
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
}

config: MaskRCNNConfig = {
    **base_config,
    "model_type": "MaskRCNN",
    "classes": 5,
    "iou_threshold": 0.1,
    "confidence_threshold": 0.1,
    "model_args": {},
    "loss_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
}


# ######### PRETRAINED WEIGHTS LOAD PATH #########
config[
    "path_model"
] = "/cluster/group/karies_2022/Simone/karies/karies-models/AAA_BYOL_test/BYOL/saved_pretrained_models/LOAD-DELETE/"

config["load_model_name"] = "pretrain_10_epochs.pth"

# ######### OTHER TRAINING PARAMS #########
freeze_backbone = True
config["batch_size"] = 4

config["name"] = "PRETRAIN-SIMONE-80epoch-batch6"

maskrcnn = MaskRCNN(config, load=True)
if freeze_backbone:
    for layer in [maskrcnn.model.transform, maskrcnn.model.backbone]:
        for weight in layer.parameters():
            weight.requires_grad = False

maskrcnn.train()
