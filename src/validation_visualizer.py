# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:57:57 2021

@author: Billy
"""

from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torchvision.models.resnet import resnet34

from l5kit.data import LocalDataManager

from l5kit.rasterization import build_rasterizer
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.geometry import transform_points
from tqdm import tqdm

import os
import cv2

from matplotlib import animation, rc
from utils.uncompressed_dataset import UncompressedAgentDataset
from utils.nll_metrics import pytorch_neg_multi_log_likelihood_batch

rc('animation', html='jshtml')

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
            nn.Linear(in_features=backbone_out_features, out_features=4096),   #out_features=4096
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)   #self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

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
    inputs = torch.from_numpy(data["image"]).unsqueeze(0).to(device)
    # Forward pass
    preds, confidences = model(inputs)
    return preds, confidences

    
# def draw_trajectory(
#         on_image: np.ndarray,
#         positions: np.ndarray,
#         rgb_color: Tuple[int, int, int],
#         radius = 1,
#         yaws: Optional[np.ndarray] = None,
# ) -> None:
#     """
#     Draw a trajectory on oriented arrow onto an RGB image
#     Args:
#         on_image (np.ndarray): the RGB image to draw onto
#         positions (np.ndarray): pixel coordinates in the image space (not displacements) (Nx2)
#         rgb_color (Tuple[int, int, int]): the trajectory RGB color
#         radius (int): radius of the circle
#         yaws (Optional[np.ndarray]): yaws in radians (N) or None to disable yaw visualisation

#     Returns: None

#     """
#     for pos in positions:
#         pred_waypoint_start = tuple(pos[:2].astype(np.int))
#         pos[0] += 1
#         pos[1] += 1
#         pred_waypoint_end = tuple(pos[:2].astype(np.int))
#         cv2.rectangle(on_image, pred_waypoint_start, pred_waypoint_end, color=rgb_color, thickness=1)


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
        'weight_path': "model_resnet34_2nd_epoch_output_cosine_annealing_5e_5_315000.pth",  #model_resnet34_unimodal_output_cosine_annealing_5e_5_310000.pth
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

val_dataset = UncompressedAgentDataset('../validation_compressed')
print(len(val_dataset))

# Semantic view
cfg['raster_params']['map_type'] = 'py_semantic'
semantic_rasterizer = build_rasterizer(cfg, dm)

# Satellite view
cfg['raster_params']['map_type'] = 'py_satellite'
satellite_rasterizer = build_rasterizer(cfg, dm)

# here I use matplotlib default colors
cmap = plt.get_cmap("tab10")
matplotlib_colors_in_rgb_int = [
    [int(255 * x) for x in cmap(i)[:3]] for i in range(10)
]


# note raster_from_agent is actually a constant matrix for each raster once you fix the raster params
raster_params = cfg['raster_params']
raster_from_agent = np.array([
    [2., 0.,  56.],
    [0., 2., 112.],
    [0., 0.,   1.],
]) if (
    raster_params['raster_size'] == [224, 224] and
    raster_params['pixel_size'] == [0.5, 0.5] and
    raster_params['ego_center'] == [0.25, 0.5]
) else None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftMultiModel(cfg)
model.to(device)
print(f'Using device {device}')
    
#load weight if there is a pretrained model
weight_path = cfg["model_params"]["weight_path"]
if weight_path:
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["model"])

if __name__ == '__main__':
    model.eval()
    torch.set_grad_enabled(False)
    #idxs = [1000, 2000, 3000, 4000]
    idxs = [5100, 7000, 7700, 8300]   #idxs = [4404, 5100, 6250, 6600, 7000, 7100, 7700, 8300, 10050]
    for splot, idx in enumerate(idxs):
        data = val_dataset[idx]
        preds, confidences = forward(data, model, device)
        preds = preds.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        world_from_agents = data["world_from_agent"]
        centroids = data["centroid"]
        agents_from_world = np.linalg.inv(world_from_agents)
        preds_in_world = np.ndarray(preds.shape)
        for idx in range(len(preds)):
            for mode in range(preds.shape[1]):   #3
                preds_in_world[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents) - centroids[:2]
        
        
        gt_csv = pd.read_csv(f'{DIR_INPUT}/scenes/validate_chopped_100/gt.csv')
        gt_row = gt_csv.loc[(gt_csv['timestamp'] == data['timestamp']) & (gt_csv['track_id'] == data['track_id'])]
        gt = gt_row.loc[:, 'coord_x00':'coord_y049'].to_numpy().reshape(50, 2)[np.newaxis, ...]
        avails = gt_row.loc[:, 'avail_0':'avail_49'].to_numpy()       
        
        # gt = data['target_positions'][np.newaxis, ...]
        # avails = data['target_availabilities'][np.newaxis, ...]
        
        loss = pytorch_neg_multi_log_likelihood_batch(torch.from_numpy(gt), torch.from_numpy(preds_in_world), torch.from_numpy(confidences), torch.from_numpy(avails))
        
        for idx in range(len(preds)):
            gt[idx, :, :] = transform_points(gt[idx, :, :] + centroids[:2], agents_from_world) 
        
        preds = preds.squeeze(0)
        confidences = confidences.squeeze(0)
        trajectories = np.concatenate((preds, gt), axis=0)
        #trajectories = trajectories[:, ::2, :]
        
        im = data['image'].transpose(1, 2, 0)
        im = semantic_rasterizer.to_rgb(im)
        im = cv2.UMat(im)
        rfg = raster_from_agent if raster_from_agent is not None else data['raster_from_agent']
        for i, coords in enumerate(trajectories):
            # target_positions_world = transform_points(coords, data['world_from_agent'])
            # target_positions_pixels = transform_points(target_positions_world, data['raster_from_world'])
            target_positions_pixels = transform_points(coords, rfg)
            if trajectories.shape[0] == 2 and i == 1:
                draw_trajectory(im, target_positions_pixels, rgb_color=matplotlib_colors_in_rgb_int[3], radius=1)
            else:
                draw_trajectory(im, target_positions_pixels, rgb_color=matplotlib_colors_in_rgb_int[i], radius=1)
        patches = [mpatches.Patch(color=cmap(m), label=f'{conf:.3f}') for m, conf in enumerate(confidences)]  
        # patches = [mpatches.Patch(color=cmap(0), label='Predicted Trajectory')]
        patches.append(mpatches.Patch(color=cmap(3), label='GT'))
        #plt.subplot(1, 4, splot+1)
        plt.figure()
        plt.axis('off')
        plt.imshow(im.get(), origin='lower')
        plt.legend(handles=patches)
        plt.title(f"NLL loss = {loss.item():.3f}")
        plt.show()


