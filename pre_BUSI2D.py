# %% import packages
import numpy as np
import os
from glob import glob
import pandas as pd

join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import cv2

# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")

# add arguments to the parser
parser.add_argument(
    "-i",
    "--busi_path",
    type=str,
    default="data/Dataset_BUSI_with_GT",
    help="path to the images",
)

parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="path to the csv file",
)

parser.add_argument(
    "-o",
    "--filtered_path",
    type=str,
    default="data/BUSI2Dtrain",
    help="path to save the npz files",
)

parser.add_argument(
    "--img_name_suffix", type=str, default=".png", help="image name suffix"
)

parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()


if not os.path.isdir(args.filtered_path):
    os.mkdir(args.filtered_path)

images_path = os.path.join(args.filtered_path,"images")
labels_path = os.path.join(args.filtered_path,"labels")

if not os.path.isdir(images_path):
    os.mkdir(images_path)

if not os.path.isdir(labels_path):
    os.mkdir(labels_path)


classes = os.listdir(args.busi_path)


for cname in tqdm(classes):
    print("Shifting class: ",cname) 
    c_path = os.path.join(args.busi_path,cname)
    c_img_names = os.listdir(c_path)
    c_img_names = sorted(c_img_names)
    mask_indices = ["_mask" in m for m in c_img_names]
    non_mask_indices = ["_mask" not in m for m in c_img_names]
    c_img_names = np.asarray(c_img_names)
    imgs = c_img_names[non_mask_indices]
    gts = c_img_names[mask_indices]

    j = 0
    for i in tqdm(range(len(imgs))):
        prefix = imgs[i].split(".")[0] + "_mask"
        current = []
        while j < len(gts) and prefix in gts[j]:
            current.append(gts[j])
            j += 1
        im = cv2.imread(os.path.join(c_path,imgs[i]))
        cv2.imwrite(os.path.join(images_path,imgs[i]),im)
        blank_image = np.zeros(im.shape[:2],dtype=np.uint8)
        for mname in current:
            mask = cv2.imread(os.path.join(c_path,mname),cv2.IMREAD_GRAYSCALE)
            blank_image += mask
        cv2.imwrite(os.path.join(labels_path,current[0]),blank_image)
    
