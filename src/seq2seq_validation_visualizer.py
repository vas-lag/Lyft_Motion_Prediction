# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:41:54 2021

@author: Billy
"""


from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torchvision.models.resnet import resnet34

from l5kit.data import LocalDataManager
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
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

# class Seq2SeqModel(nn.Module):
#     def __init__(self, num_modes=3):
#         super().__init__()
#         self.num_modes = num_modes
#         self.encoder = Encoder()
#         self.decoder = Decoder(num_modes)
        
#     def forward(self, x, device):
#         hc_n = self.encoder(x)
#         x = torch.zeros((x.shape[0], 50, 256)).to(device=device)
#         x = self.decoder(x, hc_n)
#         bs = x.shape[0]
#         x = x.view(bs, x.shape[1], self.num_modes, 2)
#         x = x.transpose(1, 2)
#         return x
        
#         # output = np.zeros((x.shape[0], 50, self.num_modes, 2))
#         # hc_n = self.encoder(x)
#         # x = torch.zeros((x.shape[0], 1, self.num_modes, 2)).to(device=device)
#         # for i in range(50):
#         #     x, hc_n = self.decoder(x, hc_n)
#         #     output[:, i] = x
#         #     x = x.view(x.shape[0], x.shape[1], -1)
#         # output = output.transpose(1, 2)
#         # return output
        
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(6, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         _, hc_n = self.lstm(x)
#         return hc_n
    
# class Decoder(nn.Module):
#     def __init__(self, num_modes):
#         super().__init__()
#         self.fc1 = nn.Linear(256, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, num_modes*2)
#         self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        
#     def forward(self, x, hc_n):
#         x, hc_n = self.lstm(x, hc_n)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
# def forward(data, model, device):
#     positions = data["history_positions"][:-1, :]
#     velocities = data["history_velocities"]
#     yaws = data["history_yaws"][:-1, :]
#     availabilities = np.expand_dims(data["history_availabilities"][:-1], -1)
#     inputs = np.expand_dims(np.concatenate((positions, velocities, yaws, availabilities), axis=-1), 0)
#     # Forward pass
#     preds = model(torch.from_numpy(inputs).to(device=device, non_blocking=True), device)
#     preds = preds
#     confidences = torch.ones((preds.shape[0], 3)).to(device=device)
#     confidences = torch.softmax(confidences, dim=1)
#     return preds, confidences




# class Seq2SeqModel(nn.Module):
#     def __init__(self, num_modes=3):
#         super().__init__()
#         self.num_modes = num_modes
#         self.encoder = Encoder()
#         self.decoder = Decoder(num_modes)
#         self.fc = nn.Linear(2*2*256, 3)
        
#     def forward(self, x, device):
#         # hc_n = self.encoder(x)
#         # x = torch.zeros((x.shape[0], 50, 256)).to(device=device)
#         # x = self.decoder(x, hc_n)
#         # bs = x.shape[0]
#         # x = x.view(bs, x.shape[1], self.num_modes, 2)
#         # x = x.transpose(1, 2)
#         # return x
        
#         output = torch.zeros((x.shape[0], 50, self.num_modes, 2)).to(device=device)
#         hc_n = self.encoder(x)
#         hc_n_t = (torch.transpose(hc_n[0], 0, 1), torch.transpose(hc_n[1], 0, 1))
#         inp = torch.flatten(torch.cat((hc_n_t[0], hc_n_t[1]), dim=1), start_dim=1)
#         confs = self.fc(inp)
#         x = torch.zeros((x.shape[0], 1, 6)).to(device=device)
#         for i in range(50):
#             x, hc_n = self.decoder(x, hc_n)
#             output[:, i] = x.view(x.shape[0], 3, 2)
#             x = x.view(x.shape[0], x.shape[1], -1)
#         # x, _ = self.decoder(x, hc_n)
#         # confs = torch.sum(x.view(x.shape[0], 3, 2), dim=-1)
#         output = output.transpose(1, 2)
#         return output, confs
        
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(6, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         _, hc_n = self.lstm(x)
#         return hc_n
    
# class Decoder(nn.Module):
#     def __init__(self, num_modes):
#         super().__init__()
#         self.fc1 = nn.Linear(256, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, num_modes*2)
#         self.lstm = nn.LSTM(input_size=6, hidden_size=256, num_layers=2, batch_first=True)
        
#     def forward(self, x, hc_n):
#         x, hc_n = self.lstm(x, hc_n)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x, hc_n





class Seq2SeqModel(nn.Module):
    def __init__(self, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.encoder = Encoder()
        self.decoder = Decoder(num_modes)
        self.probability_module = Probability_Module(num_modes)
        
    def forward(self, x, device):
        #dimension of x (bs, 7, 6)
        # hc_n = self.encoder(x)
        # x = torch.zeros((x.shape[0], 50, 256)).to(device=device)
        # x = self.decoder(x, hc_n)
        # bs = x.shape[0]
        # x = x.view(bs, x.shape[1], self.num_modes, 2)
        # x = x.transpose(1, 2)
        # return x
        

        hc_n = self.encoder(x)
        
        probability_module_input = torch.cat((hc_n[0][1], hc_n[1][1]), dim=-1)
        p = self.probability_module(probability_module_input)
        
        decoder_input = torch.zeros((x.shape[0], 50, 6)).to(device=device)
        decoder_output = self.decoder(decoder_input, hc_n)
        decoder_output = decoder_output.view(decoder_output.shape[0], decoder_output.shape[1], 3, 2)
        decoder_output = decoder_output.transpose(1, 2)
        return decoder_output, p
    
class Probability_Module(nn.Module):
    def __init__(self, num_modes):
        super().__init__()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, num_modes)
        
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
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_modes*2)
        self.lstm = nn.LSTM(input_size=6, hidden_size=256, num_layers=2, batch_first=True)
        
    def forward(self, x, hc_n):
        x, _ = self.lstm(x, hc_n)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# def forward(data, model, device):
#     positions = data["history_positions"][:-1, :]
#     velocities = data["history_velocities"]
#     yaws = data["history_yaws"][:-1, :]
#     availabilities = np.expand_dims(data["history_availabilities"][:-1], -1)
#     inputs = np.expand_dims(np.concatenate((positions, velocities, yaws, availabilities), axis=-1), 0)
#     # Forward pass
#     preds, confs = model(torch.from_numpy(inputs).to(device=device, non_blocking=True), device)
#     confidences = torch.softmax(confs, dim=1)
#     return preds, confidences


def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    positions = data["history_positions"]
    velocities = np.concatenate((data["history_velocities"], np.expand_dims(data["history_velocities"][0, :], axis=0)), axis=0)
    yaws = data["history_yaws"]
    
    #standarization
    positions[:, 0] = (positions[:, 0] + 3.4463)/35.0932
    positions[:, 1] = (positions[:, 1])/1.3746
    velocities[:, 0] = (velocities[:, 0] - 2.512)/27.1941
    velocities[:, 1] = (velocities[:, 1])/1.4085
    yaws = (yaws)/0.5561
    
    availabilities = np.expand_dims(data["history_availabilities"], axis=-1)
    inputs = np.expand_dims(np.concatenate((positions, velocities, yaws, availabilities), axis=-1), 0)
    inputs = np.flip(inputs, axis=0).copy()
    # Forward pass
    preds, confs = model(torch.from_numpy(inputs).to(device=device, non_blocking=True), device)
    confidences = torch.softmax(confs, dim=1)
    return preds, confidences


    



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
        'weight_path': "model_seq2seq_simple_prob_standarized_77500.pth",  #model_resnet34_unimodal_output_cosine_annealing_5e_5_310000.pth
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
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_mask = np.load(f"{DIR_INPUT}/scenes/validate_chopped_100/mask.npz")["arr_0"]
semantic_rasterizer = build_rasterizer(cfg, dm)
val_dataset = AgentDataset(cfg, val_zarr, semantic_rasterizer, agents_mask=val_mask)
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
model = Seq2SeqModel()
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
    #idxs = [5100, 7000, 7700, 8300]
    idxs = [12000, 13000, 14000, 15000, 16000, 17000]   #idxs = [4404, 5100, 6250, 6600, 7000, 7100, 7700, 8300, 10050]
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


