"""
NOTE:
This file is a modified copy of the original source  file:emb_CLIP.py
The original file remains unchanged.
This version was created for debugging and adaptation within the scope of the Bachelor thesis.

Edited lines of code are labeled #EDITED

Specific changes:
- Updated import to use modified CLIP utilities file (clip_debug instead of clip)
- Added custom dataset option ("FFpp_PipelineTest") for testing
- Added specific output path for train/val/test
- Added Dataset FFpp c23 and removed hardcoding of output paths so they would work with the new dataset
- Addec os import and added code to create folder and check for existing folder
- Added Dataset CelebDF
- Added Dataset FFpp c40
- Removed codeblocks for train and val split since this file is to generate embeddings for the generalization test only
"""
## conda env: clip_ex

# load packages
import torch
import clip
import tqdm
import glob
from collections import Counter
from utilities.clip_ba import compute_embeddings, compute_embeddings_batched #EDITED
import argparse
import os #EDITED

def load_dataset(dataset):
    if dataset == "cifar10":
        path = "cifar10"
    elif dataset == "cifar100":
        path = "cifar100"
    elif dataset == "ImageNet":
        path = "ImageNet/ILSVRC/Data/CLS-LOC"
    elif dataset == "ClimateTV":
        path = "ClimateTV"
    elif dataset == "CUB":
        path = "CUB_200_2011"
        # -> paths in test folder
    elif dataset == "Places365":
        path = "Places365/Data"
    elif dataset == "MiT-States":
        path = "MiT-States/release_dataset/"
    elif dataset == "ImageNet-R":
        path = "ImageNet-R/imagenet-r"
        # no train or test folder -> use val for test

#EDITED:Added my own mini FFpp dataset to test my process
    elif dataset == "FFpp_PipelineTest":
        path = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40/Test_FaceFrames"

#EDITED: Added FFpp c23 Dataset 
    elif dataset == "FFpp_c23":
        path = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c23_frames"

#EDITED: Added CelebDF Dataset
    elif dataset == "CelebDF":
        path = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/CelebDF_subset/CelebDF_frames"      
#EDITED: Added FFppc40 Dataset
    elif dataset == "FFpp_c40":
        path = "/pfs/work9/workspace/scratch/ma_ksuarezg-dcbm-ws/Test_sandbox/data/FaceForensics/c40_frames"
   
    return path

def main(dataset, emb, device):
    # load model
    device = torch.device(device)
    model, preprocess = clip.load(emb, device = device) # quickgelu is used by openai - we are using the official openai implementation
    model_name = f"CLIP-{emb}".replace("/","")
    model.eval() 
    model.to(device)

    dataset_path = load_dataset(dataset)
    output_dir = f"../../Embeddings/{dataset}" #EDITED
    os.makedirs(output_dir, exist_ok=True) #EDITED

    ## test set ##
    test_list = glob.glob(dataset_path+"/**/*.jpg", recursive=True)
    print(len(test_list))
    #output_test = f"../../Embeddings/FFpp_PipelineTest/images_{dataset}_test_{model_name}.torch" #EDITED
    output_test = f"../../Embeddings/{dataset}/images_{dataset}_test_{model_name}.torch" #EDITED
    compute_embeddings_batched(test_list, model, preprocess, output_test, device)
    output_test = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--emb', type=str, help='embedding model')
    parser.add_argument('--device', type=str, help='device')
    args = parser.parse_args()
    main(args.dataset, args.emb,  args.device)
