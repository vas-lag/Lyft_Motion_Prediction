# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 19:02:06 2021

@author: Billy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:05:38 2021

@author: Billy
"""

from typing import Dict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34

from l5kit.data import LocalDataManager

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.geometry import transform_points
from tqdm import tqdm
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, read_pred_csv, create_chopped_dataset
from l5kit.evaluation.metrics import (neg_multi_log_likelihood, time_displace, rmse, average_displacement_error_oracle, 
average_displacement_error_mean, final_displacement_error_oracle, final_displacement_error_mean, prob_true_mode)

import os
from collections import OrderedDict, defaultdict

from utils.uncompressed_dataset import UncompressedAgentDataset
from utils.metrics import (cross_track_error, along_track_error, average_cross_error_oracle, average_cross_error_mean,
                           average_along_error_oracle, average_along_error_mean)



class LyftMultiModel(nn.Module):
    def __init__(self, cfg: Dict, num_modes=3):   #num_modes=3
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        #backbone = timm.create_model(architecture, pretrained=True)
        #backbone = EfficientNet.from_pretrained('efficientnet-b1')
        self.backbone = backbone
        #print(self.backbone)

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50":
            backbone_out_features = 2048
        elif architecture == "efficientnet_b1":
            backbone_out_features = 1000
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            #nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)  

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # x = self.backbone(x)
        # x = torch.flatten(x, 1)

        x = self.head(x)
        #x = F.relu(x)
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
    
def forward(data, model, device):
    inputs = data["image"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    return preds, confidences

    
# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "../prediction_dataset_kaggle",
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 6,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'step_time' : 0.1,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'render_ego_history': True,
        'model_name': "model_resnet34_output",
        'lr': 5e-5,
        'weight_path': "model_resnet34_output_cosine_annealing_5e_5_310000.pth",
        'train': False,
        'predict': False,
        'visualize' : True
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces' : False,
        'set_origin_to_bottom' : True
    },

    'train_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 0
    },
    
    'val_data_loader': {
        'key': 'scenes/validate_chopped_100/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'max_num_steps': 50001,
        'checkpoint_every_n_steps': 10000,
    }
}

DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager()

val_cfg = cfg["val_data_loader"]
val_dataset = UncompressedAgentDataset('../validation_compressed')
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"], num_workers=val_cfg["num_workers"])
print(len(val_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftMultiModel(cfg)
model.to(device)
print(f'Using device {device}')
    
#load weight if there is a pretrained model
weight_path = cfg["model_params"]["weight_path"]
if weight_path:
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["model"])


prob_split = 0.05
bucket_count = int(1/prob_split)
count_buckets = [0] * bucket_count
found_buckets = [0] * bucket_count

model.eval()
torch.set_grad_enabled(False)

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
confidences_list = []
agent_ids = []
progress = len(val_dataloader)
# for data in progress:
data_iter = iter(val_dataloader)
for _ in tqdm(range(progress)):
    data = next(data_iter)
    preds, confidences = forward(data, model, device)
    
    # convert agent coordinates into world offsets
    preds = preds.cpu().numpy()
    world_from_agents = data["world_from_agent"].numpy()
    centroids = data["centroid"].numpy()
    coords_offset = []
    
    # convert into world coordinates and compute offsets
    for idx in range(len(preds)):
        for mode in range(3):   #3
            preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
    
    
    future_coords_offsets_pd.append(preds.copy())
    confidences_list.append(confidences.cpu().numpy().copy())
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())

pred_path = 'temp_submissions/temp_sub.csv'
write_pred_csv(pred_path,
            timestamps=np.concatenate(timestamps),
            track_ids=np.concatenate(agent_ids),
            coords=np.concatenate(future_coords_offsets_pd),
            confs = np.concatenate(confidences_list)
            )


gt = OrderedDict()
inference = OrderedDict()
metrics_dict = defaultdict(list)
for row in read_gt_csv(f'{DIR_INPUT}/scenes/validate_chopped_100/gt.csv'):
    gt[row["track_id"] + row["timestamp"]] = row
for row in read_pred_csv(pred_path):
    inference[row["track_id"] + row["timestamp"]] = row


total = len(gt.keys())
for key, ground_truth_value in tqdm(gt.items(), total=total):
    gt_coord = ground_truth_value["coord"]
    gt_avail = ground_truth_value["avail"]
    if key in inference:
        pred_coords = inference[key]["coords"]
        conf = inference[key]["conf"]
        probs_true = prob_true_mode(gt_coord, pred_coords, conf, gt_avail)
        for c in conf:
            c = int(np.floor(c * bucket_count))
            count_buckets[c] += 1
        argmax = np.argmax(probs_true)
        b = int(np.floor(conf[argmax] * bucket_count))
        found_buckets[b] += 1
            
        
df_list = [found_buckets[x] / count_buckets[x] for x in range(bucket_count)]
df = pd.DataFrame(data = (found_buckets, count_buckets, df_list))
df.to_csv(f"probability_evaluation_{cfg['model_params']['weight_path'][:-4]}.csv")


