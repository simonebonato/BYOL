from karies.config import Augmentation, ModelConfig, Task, UNetConfig
from karies.config.config_class import ModelTypes
from karies.models import U_Net_Model

base_config: ModelConfig = {
    "name": "BYOL-TEST-DELETE",
    "learning_rate": 0.0001,
    "batch_size": 8,
    "num_epochs": 80,
    "histogram_eq": True,
    "path_model": "",
    "task": Task.training,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "num_workers": 0,
    "device": "cuda",
    "image_shape": [768, 1024],
    "dataset": "dataset_4k",
    "labels_json": "labels_caries.json",
    "visualization_frequency": 1,
    "fix_random_seed": 42,
    "augmentations": [
        {"augmentation": Augmentation.gaussian_blur, "values": [9, 0.9, 1.5]},
        {"augmentation": Augmentation.rotation, "values": (-10, 10)},
        {"augmentation": Augmentation.horizontal_flip},
        {"augmentation": Augmentation.nothing},
        {"augmentation": Augmentation.elastic_transform, "values": [100.0, 10.0]},
        {"augmentation": Augmentation.random_affine, "values": [(-5, 5), (0.9, 1.1), (0, 10)]},
    ],
}

config: UNetConfig = {
    **base_config,
    "model_args": {"in_channels": 1, "classes": 1, "decoder_attention_type": None},
    "loss_func": "iou",  # "dice_ce", "gen_dice" or "iou"
    "unet_type": ModelTypes.U_Net,
}


# TODO: watch from maskrcnn
