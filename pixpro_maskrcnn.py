import sys

import torch
from karies.config import Augmentation, DatasetPartition, ModelTypes
from karies.models import MaskRCNN
from pixel_level_contrastive_learning import PixelCL
from pixpro_utils import PretrainingDataset, MaskRCNNModelWrapper
from torch.cuda.amp import GradScaler
from tqdm import tqdm

sys.path.append("/cluster/group/karies_2022/Simone/karies/karies-models/")
from configs.MaskRCNN.default_model_setting_config import model_config as config_maskrcnn

crop_size = 112  # 56. 112, 224
batch_size = 15
epochs = 10000
mixed_precision = True
device = "cuda"

config_maskrcnn["name"] = "PIXPRO-TEST-DELETE"
config_maskrcnn["batch_size"] = 10
config_maskrcnn["device"] = "cuda"
config_maskrcnn["dataset"] = "dataset_4k"
config_maskrcnn["augmentations"] = [{"augmentation": Augmentation.nothing}]
config_maskrcnn["fix_random_seed"] = 42
config_maskrcnn["image_shape"] = [768, 1024]

save_location = "/cluster/group/karies_2022/Simone/karies/karies-models/BYOL/maskrcnn_pixpro_weights/"

maskrcnn = MaskRCNN(config_maskrcnn, load=False)
wrap = MaskRCNNModelWrapper(
    maskrcnn.model.transform,
    maskrcnn.model.backbone,
)

dl = PretrainingDataset(config_maskrcnn, DatasetPartition.train, crop_size=crop_size).get_data_loaders()

learner = PixelCL(
    wrap,
    image_size = (crop_size, crop_size),
    hidden_layer_pixel = 'extra_layer',  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance = 'avgpool',     # leads to output for instance-level learning
    projection_size = 256,          # size of projection output, 256 was used in the paper
    projection_hidden_size = 2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay = 0.99,    # exponential moving average decay of target encoder
    ppm_num_layers = 1,             # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma = 2,                  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres = 0.7,           # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature = 0.3,   # temperature for the cosine similarity for the pixel contrastive loss
    alpha = 1.,                      # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro = True,               # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range = (0.6, 0.8)  # a random ratio is selected from this range for the random cutout
).cuda()

scaler = GradScaler(enabled=mixed_precision)
opt = torch.optim.Adam(learner.parameters(), lr=1e-4)
torch.cuda.empty_cache()

for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    if epoch % 500 == 0:
        model_save_name = f"maskrcnn_batch_{batch_size}_crop_{crop_size}_after_epoch_{epoch}.pth"
        torch.save(maskrcnn.model.state_dict(), save_location + model_save_name)

    for x in tqdm(dl):
        x = x.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
            loss = learner(x)  # if positive pixel pairs is equal to zero, the loss is equal to the instance level loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        learner.update_moving_average()