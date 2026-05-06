"""
NOTE:
This file is a modified copy of the original source file: dcbm_testing.py
The original file remains unchanged.
This version was created for adaptation within the scope of the bachelor thesis.

Edited lines of code are labeled #EDITED

Specific changes:
- Disabled wandb import for local execution
- Enabled training output (to_print=True)
-Switched experiment setup from CIFAR100 to FFpp_PipelineTest
- Updated import to use dcbm_ba_gen
- Adjusted paths for FF++ image and concept embeddings
- Adjusted embedding path
- Switched device to cuda
- Disabled subset-based concept selection for the FF++ pipeline test ->REVERTED for c23
- changed loading of utils to dcbm_ba_gen for generalisation test
"""

import os
import argparse
# import wandb #EDITED
import re
import clip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# add the parent directory to the path
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from utils.dcbm_ba_gen import * #EDITED to use file for generalization test


# ----------------- Constants -----------------
#embed_path = "../data/Embeddings/FFpp_PipelineTest/" #EDITED
embed_path = "../data/Embeddings/FFpp_c23/" #EDITED
#dataset = "FFpp_PipelineTest" #EDITED
dataset = "FFpp_c23" #EDITED
class_labels_path = None #EDITED
#segment_path = "../data/Segments/scripts/Seg_embs/"
segment_path = "../data/Segments/Seg_embs/" #EDITED
selected_image_concepts = "../data/Embeddings/subsets"


# ----------------- Hyperparameters -----------------
model_name = "CLIP-ViT-L14"  # "CLIP-ViT-L14", "CLIP-RN50"

cluster_method = "hierarchical"     # "hierarchical", "kmeans"
centroid_method = "mean"    # "mean", "median"
concept_per_class = None      #EDITED # How many images for each class: 5,10,20,50, None

one_hot = False
epochs = 60 #EDITED
batch_size = 16 #EDITED
crop = False                  # True without background

use_wandb = False
project = "YOUR_PROJECT_NAME"        # Define your own project name within wandb
device = "cuda:0" #EDITED

def run_training(cbm):
    """Preprocess data and train the CBM model with different hyperparameters."""
    cbm.preprocess_data(type_="standard", label_type=one_hot)
    for lambda_1, lr in [(1e-4, 1e-3)]: #EDITED
        cbm.train(
            num_epochs=epochs,
            lambda_1=lambda_1,
            lr=lr,
            device=device,
            batch_size=batch_size,
            project=project,

            to_print= True, #EDITED
            early_stopping_patience=None,
            one_hot=one_hot,
            use_wandb=use_wandb,
        )

def run_experiment(segmentation_technique, concept_name, clusters_list, load_concepts_first):
    """Run experiments with specified segmentation techniques and clustering options."""
    # Initial training without loading concepts and clustering
    if not load_concepts_first:
        cbm = CBM(embed_path, dataset, model_name, class_labels_path, device=device)
        run_training(cbm)
    # Training with loaded concepts and clustering
    for clusters in clusters_list:
        cbm = CBM(embed_path, dataset, model_name, class_labels_path, device=device)
        cbm.load_concepts(
            segment_path,
            segmentation_technique,
            concept_name,
            selected_image_concepts,
            concept_per_class,
            crop=crop,
        )
        if clusters is not None:
            cbm.cluster_image_concepts(cluster_method, clusters)
        else:
            cbm.clustered_concepts = cbm.image_segments
        cbm.centroid_concepts(centroid_method)
        run_training(cbm)

# ----------------- Experiments -----------------
experiments = [
    {
        'segmentation_technique': 'SAM2',
        'concept_name': None,
        'clusters_list': [448], #EDITED
        'load_concepts_first': True
    },
]

for exp in experiments:
    run_experiment(
        segmentation_technique=exp['segmentation_technique'],
        concept_name=exp['concept_name'],
        clusters_list=exp['clusters_list'],
        load_concepts_first=exp['load_concepts_first']
    )
