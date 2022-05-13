# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:38:29 2022

@author: Billy
"""


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info
from torchvision.models.resnet import resnet18, resnet34

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, read_pred_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace, rmse, average_displacement_error_oracle, average_displacement_error_mean, final_displacement_error_oracle, final_displacement_error_mean

import time
import random
from tqdm import tqdm
import os
from collections import OrderedDict, defaultdict
import copy
import timm

from utils.uncompressed_dataset import IndexSampler
from utils.nll_metrics import pytorch_neg_multi_log_likelihood_batch
from utils.dataset_wrapper import WrappedDataset


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
        # backbone = timm.create_model('rexnet_100', pretrained=True)
        self.backbone = backbone
        #print(backbone)
        
        num_history_channels = 14
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        

        
        #print(self.backbone)
        backbone_out_features = 512
        self.fc1 = nn.Linear(backbone_out_features, 1024)
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
    
def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    positions = data["history_positions"]
    velocities = torch.cat((data["history_velocities"][:, 0, :].unsqueeze(1), data["history_velocities"]), dim=1)
    yaws = data["history_yaws"]
    
    #standarization
    positions[:, :, 0] = (positions[:, :, 0] + 3.4463)/35.0932
    positions[:, :, 1] = (positions[:, :, 1])/1.3746
    velocities[:, :, 0] = (velocities[:, :, 0] - 2.512)/27.1941
    velocities[:, :, 1] = (velocities[:, :, 1])/1.4085
    yaws = (yaws)/0.5561
    
    availabilities = data["history_availabilities"].unsqueeze(-1)
    inputs = torch.cat((positions, velocities, yaws, availabilities), axis=-1).to(device=device, non_blocking=True)
    inputs = torch.flip(inputs, (1,))
    target_availabilities = data["target_availabilities"].to(device=device, non_blocking=True)
    targets = data["target_positions"].to(device=device, non_blocking=True)
    image = data['image'].to(device)

    # Forward pass
    preds, confs = model(inputs, image, device)
    #confidences = torch.ones((preds.shape[0], 3)).to(device=device)
    confidences = torch.softmax(confs, dim=1)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences
        

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
        'model_name': "model_seq2seq_cnn_full_zeros_standarized",
        'lr': 1e-4,        #5e-5,
        'weight_path': "model_seq2seq_cnn_full_zeros_standarized_310000.pth",
        'train': False,
        'predict': True,
        'validate': True,
        'visualize' : False
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
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
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
        'max_num_steps': 10_001,
        'checkpoint_every_n_steps': 10_000,
        'validate_every_n_steps': 10_000
    }
}

rast_cfg = copy.deepcopy(cfg)
rast_cfg["model_params"]["history_num_frames"] = 6


def my_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize_worker(worker_id)

if __name__ == '__main__':
    random.seed(42) 
    
    # set env variable for data
    DIR_INPUT = cfg["data_path"]
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    
    # ===== INIT TRAIN DATASET============================================================
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(rast_cfg, dm)
    train_dataset = WrappedDataset()
    print(f'{len(train_dataset) = }')
    index = list(range(len(train_dataset)))
    random.shuffle(index)
    start_index = 300_000
    index = index[start_index * train_cfg["batch_size"]:]
    train_dataloader = DataLoader(train_dataset, sampler=IndexSampler(index), shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                 num_workers=train_cfg["num_workers"], pin_memory=True, worker_init_fn=my_init_fn)
    print("==================================TRAIN DATA==================================")
    # print(train_dataset)
    
    
    #====== INIT TEST DATASET=============================================================
    test_cfg = cfg["test_data_loader"]
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
                                  num_workers=test_cfg["num_workers"], pin_memory=True)
    print("==================================TEST DATA==================================")
    
    #====== INIT VALID DATASET============================================================
    val_cfg = cfg["val_data_loader"]
    val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
    val_mask = np.load(f"{DIR_INPUT}/scenes/validate_chopped_100/mask.npz")["arr_0"]
    val_dataset = AgentDataset(cfg, val_zarr, rasterizer, agents_mask=val_mask)
    val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                                num_workers=val_cfg["num_workers"], pin_memory=True)
    
    
    # # ==== INIT MODEL=================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqModel()
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)  ####5000
    
    print(f'Using device {device}')
    
    #load weight if there is a pretrained model
    weight_path = cfg["model_params"]["weight_path"]
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
            
            loss, _, _ = forward(data, model, device)
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
                torch.save(state, f'{model_name}_{i}.pth')
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
                progress = 1500  #len(val_dataloader)
                # for data in progress:
                data_iter = iter(val_dataloader)
                for _ in range(progress):
                    data = next(data_iter)
                    _, preds, confidences = forward(data, model, device)
                    
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
        if start_index == 0:   ##############0
            df.to_csv(f"train_losses_{model_name}.csv", index=False)
            results.to_csv(f"train_metrics_{model_name}.csv", index=False)
        else:
            df.to_csv(f"train_losses_{model_name}.csv", mode='a', index=False, header=False)
            results.to_csv(f"train_metrics_{model_name}.csv", mode='a', index=False, header=False)
        print(f"Total training time is {(time.time()-start)/60} mins")
        print(results)
        
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
            
            _, preds, confidences = forward(data, model, device)
        
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
        pred_path = f'submission_{cfg["model_params"]["weight_path"][:-4]}.csv'
        write_pred_csv(pred_path,
                    timestamps=np.concatenate(timestamps),
                    track_ids=np.concatenate(agent_ids),
                    coords=np.concatenate(future_coords_offsets_pd),
                    confs = np.concatenate(confidences_list)
                  )

