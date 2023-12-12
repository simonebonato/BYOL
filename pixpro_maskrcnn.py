import torch
import torchvision.transforms as T
from byol_pretrain.BYOL_MaskRCNN import MaskRCNNModelWrapper
from karies.config import MaskRCNNConfig, ModelConfig, ModelTypes, Task
from karies.models import MaskRCNN
from pixel_level_contrastive_learning import PixelCL
from torch.cuda.amp import GradScaler
from torchvision.models import resnet50
from tqdm import tqdm

base_config: ModelConfig = {
    "name": "PIXPRO-TEST-DELETE",
    "task": Task.training,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 2,
    "num_workers": 0,
    "device": "cuda",
    "path_model": "/cluster/group/karies_2022/Simone/karies/karies-models/AAA_BYOL_test/BYOL/saved_pretrained_models/maskrcnn-weekend-test-batch-6/",
    "load_model_name": "pretrain_80_epochs.pth",
    "num_epochs": 100,
    "image_shape": [768, 768],
    "dataset": "dataset_4k",
    "labels_json": "labels_caries.json",
    "histogram_eq": False,
    "visualization_frequency": 1,
    "augmentations": [],
    "fix_random_seed": 42,
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

m = MaskRCNN(config, load=False)

wrap = MaskRCNNModelWrapper(
    m.model.transform,
    m.model.backbone,
)

learner = PixelCL(
    wrap,
    image_size=(768, 768),
    hidden_layer_pixel="backbone.body.layer4",  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance=-1,  # leads to output for instance-level learning
    projection_size=256,  # size of projection output, 256 was used in the paper
    projection_hidden_size=2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay=0.99,  # exponential moving average decay of target encoder
    ppm_num_layers=1,  # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma=2,  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres=0.7,  # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature=0.3,  # temperature for the cosine similarity for the pixel contrastive loss
    alpha=1.0,  # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro=True,  # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range=(0.6, 0.8),  # a random ratio is selected from this range for the random cutout
).cuda()


scaler = GradScaler()
opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

torch.cuda.empty_cache()

loader, _ = m.get_data_loaders()

for epoch in range(1000):
    for x, y in tqdm(loader):
        x = torch.stack(x).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = learner(x)  # if positive pixel pairs is equal to zero, the loss is equal to the instance level loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        learner.update_moving_average()  # update moving average of target encoder

    if epoch % 100 == 0:
        torch.save(
            m.model.state_dict(),
            f"/cluster/group/karies_2022/Simone/karies/karies-models/AAA_BYOL_test/BYOL/maskrcnn_pixpro_weights/pretrained_weights_{epoch}_epochs.pth",
        )
