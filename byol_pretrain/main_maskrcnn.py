import sys

sys.path.append("/cluster/group/karies_2022/Simone/karies/karies-models")

from pathlib import Path

from byol_pretrain.BYOL_MaskRCNN import BYOL_MaskRCNN_Class, MaskRCNNModelWrapper
from configs.MaskRCNN import default_model_setting_config
from karies.models import MaskRCNN

config = default_model_setting_config.model_config

# hyperparameters
pretrain_epochs = 1000
log_dir = Path("./logs/epochs-1000_batch-6")
device = "cuda"
lin_evaluation_frequency = None  # Not needed in MaskRCNN

config["batch_size"] = 6

#  define the model and the loaders
maskrcnn = MaskRCNN(config, False)
loader, _ = maskrcnn.get_data_loaders()
wrap = MaskRCNNModelWrapper(maskrcnn.model.transform, maskrcnn.model.backbone)

byol = BYOL_MaskRCNN_Class(
    wrap,
    loader,
    None,
    pretrain_epochs,
    log_dir,
    input_dims=3,
    img_dims=(700, 700),
    hidden_features=4096,
    device=device,
    lin_evaluation_frequency=lin_evaluation_frequency,
    mixed_precision_enabled=True,
    original_maskrcnn=maskrcnn,
    save_weights_every=40,
)


byol.pretrain()
