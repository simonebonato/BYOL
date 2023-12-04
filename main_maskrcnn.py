import sys

from karies.models import MaskRCNN

sys.path.append("/cluster/group/karies_2022/Simone/karies/karies-models")
from pathlib import Path

from byol_pretrain.BYOL_MaskRCNN import BYOL_MaskRCNN, MaskRCNNModelWrapper
from configs.MaskRCNN import default_model_setting_config

config = default_model_setting_config.model_config
maskrcnn = MaskRCNN(config, False)

loader, val_loader = maskrcnn.get_data_loaders()

wrap = MaskRCNNModelWrapper(maskrcnn.model.transform, maskrcnn.model.backbone, True)

from pathlib import Path

pretrain_epochs = 1000
log_dir = Path("./logs/maskrcnn-byol-long-BATCH8")
device = "cuda"
lin_evaluation_frequency = None

byol = BYOL_MaskRCNN(
    wrap,
    loader,
    val_loader,
    pretrain_epochs,
    log_dir,
    input_dims=3,
    img_dims=(700, 700),
    hidden_features=2048,
    device=device,
    lin_evaluation_frequency=lin_evaluation_frequency,
    mixed_precision_enabled=True,
    original_maskrcnn=maskrcnn,
    save_weights_every=20,
)


byol.pretrain()
