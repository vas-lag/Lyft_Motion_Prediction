# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:36:31 2021

@author: Billy
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from tqdm import tqdm
import pandas as pd
import timm
from efficientnet_pytorch import EfficientNet

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, read_pred_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace, rmse, average_displacement_error_oracle, average_displacement_error_mean, final_displacement_error_oracle, final_displacement_error_mean
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from pathlib import Path
from collections import OrderedDict, defaultdict

import os
import time 
import random 

from utils.nll_metrics import pytorch_neg_multi_log_likelihood_batch
from utils.ade_metrics import pytorch_average_displacement_error_batch, pytorch_uber_paper_error_batch
from utils.uncompressed_dataset import UncompressedAgentDataset, IndexSampler


class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):   #num_modes=3
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
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
        elif architecture == "efficientnet_b0":
            backbone_out_features = 1280
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
        

        x = self.head(x)
        #x = F.relu(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        return x
    
def forward(data, model, feature_extractor1, feature_extractor2, device, mode='train', criterion = pytorch_neg_multi_log_likelihood_batch):  #  pytorch_average_displacement_error_batch
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device=device, non_blocking=True)
    targets = data["target_positions"].to(device=device, non_blocking=True)
    # Forward pass
    torch.set_grad_enabled(False)
    features1 = feature_extractor1(inputs)
    features2 = feature_extractor2(inputs)
    features = torch.cat((features1, features2), dim=1)
    if mode == 'train':
        torch.set_grad_enabled(True)
    preds, confidences = model(features)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences



class HeadModel(nn.Module):
    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()
        
        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096*2, out_features=self.num_preds + num_modes) 

    def forward(self, x):
        x = self.logit(x)
        
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
        


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
        'model_name': "model_ensemble_output",   
        'scheduler_name': "cosine_annealing_5e_5",
        'lr': 5e-5,
        'weight_path': "model_ensemble_output_cosine_annealing_5e_5_80000.pth", 
        'train': True,
        'predict': False,
        'validate': True
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
        'key': 'scenes/train.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 4
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 0
    },
    
    'val_data_loader': {
        'key': 'scenes/validate_chopped_100/validate.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'max_num_steps': 70_001,
        'checkpoint_every_n_steps': 10_000,
        'validate_every_n_steps': 10_000
    }
}

if __name__ == "__main__":
    random.seed(42)  
    
    # set env variable for data
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    
    # ===== INIT TRAIN DATASET============================================================
    train_cfg = cfg["train_data_loader"]
    train_dataset = UncompressedAgentDataset('D:/lyft_compressed_npz_frames')
    print(f'{len(train_dataset) = }')
    index = list(range(len(train_dataset)))
    random.shuffle(index)
    start_index = 80_000
    index = index[start_index * train_cfg["batch_size"]:]
    train_dataloader = DataLoader(train_dataset, sampler=IndexSampler(index), shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                 num_workers=train_cfg["num_workers"], pin_memory=True)
    print("==================================TRAIN DATA==================================")
    # print(train_dataset)
    
    
    #====== INIT TEST DATASET=============================================================
    test_cfg = cfg["test_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
                                  num_workers=test_cfg["num_workers"])
    print("==================================TEST DATA==================================")
    print(test_dataset)
    
    #====== INIT VALID DATASET============================================================
    
    val_dataset = UncompressedAgentDataset('../validation_compressed')
    val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                                num_workers=val_cfg["num_workers"])
    
    
    # # ==== INIT MODEL=================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = LyftMultiModel(cfg)
    cfg["model_params"]["model_architecture"] = 'resnet18'
    model2 = LyftMultiModel(cfg)
    model = HeadModel(cfg)
    
    model1.to(device)
    model2.to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2500, T_mult=2)   #best lr is 5e-5
    print(f'Using device {device}')
    
    #load weight if there is a pretrained model
    weight_path1 = "model_resnet34_2nd_epoch_output_cosine_annealing_5e_5_315000.pth"
    weight_path2 = "model_resnet18_2nd_epoch_output_cosine_annealing_5e_5_157500.pth"
    weight_path = cfg['model_params']['weight_path']
    
    checkpoint1 = torch.load(weight_path1)
    model1.load_state_dict(checkpoint1["model"])
    
    checkpoint2 = torch.load(weight_path2)
    model2.load_state_dict(checkpoint2["model"])
    
    if weight_path:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    # ==== TRAINING LOOP =========================================================
    if cfg["model_params"]["train"]:
        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(start_index, start_index + cfg["train_params"]["max_num_steps"]))
        num_iter = cfg["train_params"]["max_num_steps"]
        losses_train = []
        losses_validate = []
        avg_losses_train  = 0
        avg_losses_train_1k = 20 * [1000]
        iterations = []
        train_metrics = []
        validate_metrics = []
        times = []
        model_name = cfg["model_params"]["model_name"]
        scheduler_name = cfg["model_params"]["scheduler_name"]
        start = time.time()
        cumulative_loss = 0
        for i in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            
            loss, _, _ = forward(data, model, model1, model2, device)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            cumulative_loss += loss.detach()
            
            scheduler.step()
            
            if i % 50 == 0 and i > start_index:
                losses_train.append(cumulative_loss.item() / 50)
                cumulative_loss = 0
                avg_losses_train = losses_train[-1]
                avg_losses_train_1k[int(i / 50) % 20] = losses_train[-1]
                progress_bar.set_description(f"loss(avg): {np.mean(losses_train)}, average of last 50 losses: {avg_losses_train}, average of last 1k losses: {np.mean(avg_losses_train_1k)}")
                
            if (i - start_index) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and i > start_index:
                state = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
                torch.save(state, f'{model_name}_{scheduler_name}_{i}.pth')
                iterations.append(i)
                train_metrics.append(np.mean(losses_train))
                times.append((time.time()-start)/60)
                
            if (i - start_index) % cfg['train_params']['validate_every_n_steps'] == 0 and i > start_index and cfg["model_params"]["validate"]: 
                print("validating...")
                model.eval()
                torch.set_grad_enabled(False)
                
                # store information for evaluation
                future_coords_offsets_pd = []
                timestamps = []
                confidences_list = []
                agent_ids = []
                progress = 750  #len(val_dataloader)
                # for data in progress:
                data_iter = iter(val_dataloader)
                for _ in range(progress):
                    data = next(data_iter)
                    _, preds, confidences = forward(data, model, model1, model2, device, 'validate')
                    
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
                
                pred_path = 'temp_submissions/temp_sub2.csv'
                write_pred_csv(pred_path,
                            timestamps=np.concatenate(timestamps),
                            track_ids=np.concatenate(agent_ids),
                            coords=np.concatenate(future_coords_offsets_pd),
                            confs = np.concatenate(confidences_list)
                            )
                # metrics = compute_metrics_csv(f'{DIR_INPUT}/scenes/validate_chopped_100/gt.csv', pred_path, [neg_multi_log_likelihood, time_displace])
                # for metric_name, metric_mean in metrics.items():
                #     print(metric_name, metric_mean)
                
                gt = OrderedDict()
                inference = OrderedDict()
                metrics_dict = defaultdict(list)
                for row in read_gt_csv(f'{DIR_INPUT}/scenes/validate_chopped_100/gt.csv'):
                    gt[row["track_id"] + row["timestamp"]] = row
                for row in read_pred_csv(pred_path):
                    inference[row["track_id"] + row["timestamp"]] = row
                metrics = [neg_multi_log_likelihood, average_displacement_error_oracle, average_displacement_error_mean, final_displacement_error_oracle, final_displacement_error_mean]
                
                for key, ground_truth_value in gt.items():
                    gt_coord = ground_truth_value["coord"]
                    gt_avail = ground_truth_value["avail"]
                    if key in inference:
                        pred_coords = inference[key]["coords"]
                        conf = inference[key]["conf"]
                        for metric in metrics:
                            metrics_dict[metric.__name__].append(metric(gt_coord, pred_coords, conf, gt_avail))
                val_metric = np.mean(metrics_dict[neg_multi_log_likelihood.__name__], axis=0)
                validate_metrics.append(val_metric)
                print({metric_name: np.mean(values, axis=0) for metric_name, values in metrics_dict.items()})
                
        
        train_index = np.arange(start_index + 50, start_index + cfg["train_params"]["max_num_steps"], 50)   
        df = pd.DataFrame({'iterations': train_index, 'train loss': losses_train})
        results = pd.DataFrame({'iterations': iterations, 'metrics (avg)': train_metrics, 'validation metrics': validate_metrics, 'elapsed_time (mins)': times})
        if start_index == 10_000:
            df.to_csv(f"train_losses_{model_name}_{scheduler_name}.csv", index=False)
            results.to_csv(f"train_metrics_{model_name}_{scheduler_name}.csv", index=False)
        else:
            df.to_csv(f"train_losses_{model_name}_{scheduler_name}.csv", mode='a', index=False, header=False)
            results.to_csv(f"train_metrics_{model_name}_{scheduler_name}.csv", mode='a', index=False, header=False)
        print(f"Total training time is {(time.time()-start)/60} mins")
        print(results.head())
    
    # ==== EVAL LOOP ================================================================
    if cfg["model_params"]["predict"]:
        
        model.eval()
        torch.set_grad_enabled(False)
    
        # store information for evaluation
        future_coords_offsets_pd = []
        timestamps = []
        confidences_list = []
        agent_ids = []
    
        progress_bar = tqdm(test_dataloader)
        
        for data in progress_bar:
            
            _, preds, confidences = forward(data, model, model1, model2, device, 'predict')
        
            #fix for the new environment
            preds = preds.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []
            
            # convert into world coordinates and compute offsets
            for idx in range(len(preds)):
                for mode in range(3):
                    preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
        
            future_coords_offsets_pd.append(preds.copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())
                    
        #create submission to submit to Kaggle
        pred_path = 'submission.csv'
        write_pred_csv(pred_path,
                    timestamps=np.concatenate(timestamps),
                    track_ids=np.concatenate(agent_ids),
                    coords=np.concatenate(future_coords_offsets_pd),
                    confs = np.concatenate(confidences_list)
                  )
        

        
    



