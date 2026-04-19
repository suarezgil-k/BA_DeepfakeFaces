"""
NOTE:
This file is a modified copy of the original source file: create_segments_SAM2.py
The original file remains unchanged.
This version was created for adaptation within the scope of the bachelor thesis.
Edited lines of code are labeled #EDITED

Specific changes:
- Replaced dataset loading with custom data_loading_ba to support FF++ pipeline
- Adjusted SAM2 import to use the installed sam2 package instead of a missing local utilities path
- Corrected SAM2 config path (sam2_1/sam2.1 -> sam2.1/sam2.1)
- Changed sam2_checkpoint path to match with folder structure
"""

## conda env: sam2

## import packages
import numpy as np
import os
import torch
import torch.multiprocessing as mp
import glob
import tqdm
import argparse

from sam2.build_sam import build_sam2 #EDITED
from utilities.data_loading_ba import load_data #EDITED
from utilities.SAM2 import generate_mask

torch.cuda.empty_cache() 

def load_model(device):      
    sam2_checkpoint = "utilities/segment_anything_2/checkpoints/sam2.1_hiera_large.pt" #EDITED
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" #EDITED

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2 = sam2.to(device)
    return sam2

# Define multiprocessing worker
def main(dataset, create_segments, create_stats, device, check_existing):
    # load data
    prompt_specs = None
    train_paths, dataset_name = load_data(dataset,"SAM2", prompt_specs, check_existing)
    out_path = f"{dataset_name}_SAM2"

    # create outdir if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    sam2 = load_model(device)

    stats = {}

    # accumulate stats
    total_no_masks_count = 0
    total_img_wo_masks = []

    for img_path in tqdm.tqdm(train_paths):
        s, ipath, no_masks_count, img_wo_masks = generate_mask(sam2, img_path, out_path, create_stats, create_segments, dataset)
        stats[ipath] = s

        total_no_masks_count += no_masks_count
        total_img_wo_masks += img_wo_masks

    print("Mask generation completed.")
    print(f"Total stats collected: {len(stats)}")

    print("Total no masks count:", total_no_masks_count)
    
    stats["total_no_masks_count"] = total_no_masks_count
    stats["total_img_wo_masks"] = total_img_wo_masks
    # save stats
    if os.path.exists(f"stats/{dataset}_SAM2_STATS.torch"):
        old_stats = torch.load(f"stats/{dataset}_SAM2_STATS.torch")
        stats.update(old_stats)   
    torch.save(stats, f"stats/{dataset}_SAM2_STATS.torch")
    print(len(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create segments using SAM2")
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to use. Please select from CUB, ImageNet, ImageNette, ImageWoof')
    parser.add_argument('--create_segments', type=bool, required=False, default=True, help='Whether to create segment images or not')
    parser.add_argument('--create_stats', type=bool, required=False, default=True, help='Whether to create stats or not')
    parser.add_argument('--device', type=str, required=False, default="cuda:0", help='The device to use for detection')
    parser.add_argument('--check_existing', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to check if the image has already been segmented')
    args = parser.parse_args()
    
    main(args.dataset, args.create_segments, args.create_stats, args.device, args.check_existing)
