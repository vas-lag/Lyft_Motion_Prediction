# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 20:21:57 2021

@author: Billy
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

csv_path = 'probability_evaluation_model_resnet34_output_cosine_annealing_5e_5_310000.csv'
prob_split = 0.05

plt.rcParams.update({'font.size': 22})
df = pd.read_csv(csv_path)
probs = df.iloc[2][1:]
conf_start = prob_split / 2
bucket_counter = int(1/prob_split)
confs = [conf_start + i * prob_split for i in range(bucket_counter)]
plt.figure()
plt.plot(confs, probs, color='steelblue', linewidth=2.5)
x = np.linspace(0, 1, 400)
plt.plot(x, x, linestyle='--', color='gray', linewidth=2)
plt.xlabel('predicted mode probability')
plt.ylabel('Best mode likelihood')

