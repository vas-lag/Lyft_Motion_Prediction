# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:53:41 2022

@author: Billy
"""


from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet34

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.geometry import transform_points
from tqdm import tqdm

import os
import cv2
import copy

from matplotlib import animation, rc
from utils.uncompressed_dataset import UncompressedAgentDataset
from utils.nll_metrics import pytorch_neg_multi_log_likelihood_batch

from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace_all_modes

rc('animation', html='jshtml')

class Seq2SeqModel(nn.Module):
    def __init__(self, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.encoder = Encoder()
        self.decoder = Decoder(num_modes)
        self.probability_module = Probability_Module(num_modes)
        self.cnn = CNNModel()
        
    def forward(self, x, im, device):
        hc_n = self.encoder(x)
        
        cnn_features = self.cnn(im)
        
        probability_module_input = torch.cat((hc_n[0][1], hc_n[1][1], cnn_features), dim=-1)
        p = self.probability_module(probability_module_input)
        
        hc_n_00 = torch.cat((hc_n[0][0], cnn_features), dim=-1)
        hc_n_01 = torch.cat((hc_n[0][1], cnn_features), dim=-1)
        hc_n_10 = torch.cat((hc_n[1][0], cnn_features), dim=-1)
        hc_n_11 = torch.cat((hc_n[1][1], cnn_features), dim=-1)
        hn = torch.cat((hc_n_00.unsqueeze(0), hc_n_01.unsqueeze(0)), dim=0)
        cn = torch.cat((hc_n_10.unsqueeze(0), hc_n_11.unsqueeze(0)), dim=0)
        
        decoder_input = torch.zeros((x.shape[0], 50, 6)).to(device=device)
        decoder_output = self.decoder(decoder_input, (hn, cn))
        decoder_output = decoder_output.view(decoder_output.shape[0], decoder_output.shape[1], 3, 2)
        decoder_output = decoder_output.transpose(1, 2)
        return decoder_output, p
    
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone
        #print(self.backbone)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
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
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    
class Probability_Module(nn.Module):
    def __init__(self, num_modes):
        super().__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_modes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        _, hc_n = self.lstm(x)
        return hc_n
    
class Decoder(nn.Module):
    def __init__(self, num_modes):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_modes*2)
        self.lstm = nn.LSTM(input_size=6, hidden_size=768, num_layers=2, batch_first=True)
        
    def forward(self, x, hc_n):
        x, _ = self.lstm(x, hc_n)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def forward(data, model, device):
    positions = torch.from_numpy(data["history_positions"]).unsqueeze(0)
    #velocities = torch.cat((torch.from_numpy(data["history_velocities"]), torch.from_numpy(data["history_velocities"][0, :]).unsqueeze(0)), dim=0).unsqueeze(0)
    velocities = torch.cat((torch.from_numpy(data["history_velocities"]), torch.from_numpy(data["history_velocities"][0, :]).unsqueeze(0)), dim=0).unsqueeze(0)
    yaws = torch.from_numpy(data["history_yaws"]).unsqueeze(0)
    
    # #standarization
    # positions[:, :, 0] = (positions[:, :, 0] + 3.4463)/35.0932
    # positions[:, :, 1] = (positions[:, :, 1])/1.3746
    # velocities[:, :, 0] = (velocities[:, :, 0] - 2.512)/27.1941
    # velocities[:, :, 1] = (velocities[:, :, 1])/1.4085
    # yaws = (yaws)/0.5561
    
    availabilities = torch.from_numpy(data["history_availabilities"]).unsqueeze(-1).unsqueeze(0)
    inputs = torch.cat((positions, velocities, yaws, availabilities), axis=-1).to(device=device, non_blocking=True)
    inputs = torch.flip(inputs, (1,))
    image = torch.from_numpy(data["image"][-3:, :, :]).unsqueeze(0).to(device, non_blocking=True)
    #image = torch.from_numpy(data["image"]).unsqueeze(0).to(device, non_blocking=True)
    # Forward pass
    preds, confs = model(inputs, image, device)
    confidences = torch.softmax(confs, dim=1)
    return preds, confidences
    
# def forward(data, model, device):
#     inputs = torch.from_numpy(data["image"]).unsqueeze(0).to(device)
#     # Forward pass
#     preds, confidences = model(inputs)
#     return preds, confidences

    
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
        'history_num_frames': 25,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'step_time' : 0.1,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'render_ego_history': True,
        'model_name': "model_resnet34_output",
        'lr': 5e-5,
        'weight_path': "model_seq2seq_cnn_zeros_310000.pth",  
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

rast_cfg = copy.deepcopy(cfg)
rast_cfg["model_params"]["history_num_frames"] = 6


DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager()

rasterizer = build_rasterizer(rast_cfg, dm)

val_cfg = cfg["val_data_loader"]
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_mask = np.load(f"{DIR_INPUT}/scenes/validate_chopped_100/mask.npz")["arr_0"]
val_dataset = AgentDataset(cfg, val_zarr, rasterizer, agents_mask=val_mask)

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
    [0., -2., 112.],
    [0., 0.,   1.],
]) if (
    raster_params['raster_size'] == [224, 224] and
    raster_params['pixel_size'] == [0.5, 0.5] and
    raster_params['ego_center'] == [0.25, 0.5]
) else None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Seq2SeqModel()
#model = LyftMultiModel(cfg)
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
    #idxs = [5100, 7000, 7700, 8300, 8330, 11300, 15300]
    #idxs = [10300, 11300, 12300, 13300, 14300, 15300, 16300]   
    #idxs = [ 5100, 6250, 6600, 7000, 7100, 7700, 8300, 10050]
    #idxs = [2474, 1034, 4823, 3658, 2732, 4822, 1938, 985]
    #idxs = [18055, 75438, 92376, 25076, 2346, 7526, 7000, 7700, 8330, 15300, 3729]
    #idxs = [7700, 3729, 83872, 84828]
    #comparison_idxs = [7000, 3729, 10300]
    cnn_idxs = [7000, 7700, 15300, 1034, 4823, 985]
    hybrid_idxs = [10050, 9476, 63546, 36581, 23913, 3729, 37944, 9363, 36281]
    hard_idxs = [18055, 25076, 2346, 8330]
    for splot, idx in enumerate(hard_idxs):
        data = val_dataset[idx]
        preds, confidences = forward(data, model, device)
        preds = preds.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        world_from_agents = data["world_from_agent"]
        centroids = data["centroid"]
        agents_from_world = np.linalg.inv(world_from_agents)
        preds_in_world = np.ndarray(preds.shape)
        for ix in range(len(preds)):        #batch
            for mode in range(preds.shape[1]):   #3
                preds_in_world[ix, mode, :, :] = transform_points(preds[ix, mode, :, :], world_from_agents) - centroids[:2]
        
        
        gt_csv = pd.read_csv(f'{DIR_INPUT}/scenes/validate_chopped_100/gt.csv')
        gt_row = gt_csv.loc[(gt_csv['timestamp'] == data['timestamp']) & (gt_csv['track_id'] == data['track_id'])]
        gt = gt_row.loc[:, 'coord_x00':'coord_y049'].to_numpy().reshape(50, 2)
        avails = gt_row.loc[:, 'avail_0':'avail_49'].to_numpy()       
        
        
        #loss = pytorch_neg_multi_log_likelihood_batch(torch.from_numpy(gt).unsqueeze(0), torch.from_numpy(preds_in_world), torch.from_numpy(confidences), torch.from_numpy(avails))
        nll_loss = neg_multi_log_likelihood(gt, preds_in_world[0], confidences[0], avails[0])
        all_displacement_metrics = time_displace_all_modes(gt, preds_in_world[0], confidences[0], avails[0])
        
        
        gt = transform_points(gt + centroids[:2], agents_from_world) 
        
        preds = preds.squeeze(0)
        confidences = confidences.squeeze(0)
        trajectories = np.concatenate((preds, np.expand_dims(gt, axis=0)), axis=0)
        # trajectories = np.expand_dims(gt, axis=0)
        
        trajectories = trajectories[:, avails[0] != 0, :]
        #trajectories = trajectories[:, ::2, :]
        
        im = data['image'].transpose(1, 2, 0)
        im = semantic_rasterizer.to_rgb(im)
        im = cv2.UMat(im)
        #rfg = raster_from_agent if raster_from_agent is not None else data['raster_from_agent']
        rfg = data['raster_from_agent']
        for i, coords in enumerate(trajectories):
            # target_positions_world = transform_points(coords, data['world_from_agent'])
            # target_positions_pixels = transform_points(target_positions_world, data['raster_from_world'])
            target_positions_pixels = transform_points(coords, rfg)
            if trajectories.shape[0] == 2 and i == 1:
                draw_trajectory(im, target_positions_pixels, rgb_color=matplotlib_colors_in_rgb_int[3], radius=1)
            else:
                draw_trajectory(im, target_positions_pixels, rgb_color=matplotlib_colors_in_rgb_int[i], radius=1)
                #draw_trajectory(im, target_positions_pixels, rgb_color=matplotlib_colors_in_rgb_int[3], radius=1)
        patches = [mpatches.Patch(color=cmap(m), label=f'{conf:.3f}') for m, conf in enumerate(confidences)]  
        # patches = [mpatches.Patch(color=cmap(0), label='Predicted Trajectory')]
        patches.append(mpatches.Patch(color=cmap(3), label='GT'))
        #plt.subplot(2, 2, splot+1)
        plt.figure()
        #plt.subplot(2, 2, splot + 1)
        plt.axis('off')
        
        border_image = np.zeros((226, 226, 3), dtype=np.uint8)
        border_image[1:-1, 1:-1, :] = im.get()
        
        plt.imshow(border_image, origin='lower')
        #plt.imshow(im.get(), origin='lower')
        plt.legend(handles=patches, prop={'size': 22})
        plt.title(f"NLL loss = {nll_loss[0].item():.2f} \nADE = {all_displacement_metrics[3]:.3f}", size=28)
        plt.show()


