import argparse
from cmath import pi
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os 
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.general import increment_path

from td3_agent import TD3
from dataset import OD_Dataset
from detect import yolo
from reinforcement import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='coco')
    parser.add_argument('--project', default='runs/score', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Result directary
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)  # increment run
    writer = SummaryWriter(save_dir + "/metrics")

    # yolov7 model
    yolo_model = yolo(save_dir=save_dir)

    # Create td3 model
    action_dim = 4
    weight_dir = "runs/train/exp4"
    agent = TD3(action_dim)
    agent.load(weight_dir)
    agent.actor.eval()

    # Trainloader
    trainDataset = OD_Dataset(opt.dataset_path, mode='test')
    trainDataloader = DataLoader(trainDataset, batch_size=1, shuffle=True)

    # start scoring
    precision, recall = [], []
    rl_precision, rl_recall = [], []
    original_score_list, adjusted_score_list = [], []
    score_diff_list = []
    action_list = []
    update_cnt = 0
    pbar = tqdm(enumerate(trainDataloader), total=len(trainDataloader))
    for i, data in pbar:
        '''
            img     : (3, height, width)
            target  : [label, xmin, ymin, xmax, ymax]
        '''
        img, target, img_path = data
        img = img.squeeze(dim=0)
        target = target.squeeze(dim=0)
        labels, boxs = target[:, 0], target[:, 1:]

        # Origin Image
        precision_origin, recall_origin = yolo_model.detectImg(img, target)
        Origin_score = get_score(precision_origin, recall_origin)

        # RL image
        action = agent.select_action(img.unsqueeze(dim=0))[0] 
        trans_action = transform_action(action, 0.5, 1.5)
        adjust_img = modify_image(img, *trans_action)
        precision_RL, recall_RL = yolo_model.detectImg(adjust_img, target)
        RL_score = get_score(precision_RL, recall_RL)

        score_diff = RL_score-Origin_score

        # append the metric to list
        precision.append(precision_origin)
        recall.append(recall_origin)
        rl_precision.append(precision_RL)
        rl_recall.append(recall_RL)

        # append the score to list
        original_score_list.append(Origin_score)
        adjusted_score_list.append(RL_score)

        # append the action to list
        action_list.append(trans_action)

        # Writer & Record
        writer.add_scalar('score',score_diff, update_cnt)
        score_diff_list.append(score_diff)

        update_cnt += 1

    # End Scoring ---------------------------------------------------------

    # store in the csv
    score_diff_pd = pd.DataFrame(score_diff_list, columns=['score_gap'])
    action_pd = pd.DataFrame(action_list, columns=['brightness', 'saturation', 'contrast', 'sharpness'])
    all_pd = pd.concat([score_diff_pd, action_pd], axis=1)

    all_pd.to_csv(os.path.join(save_dir, "metric.csv"), index=False)    # save metric

    # mean & std calculation
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    rl_precision_mean = np.mean(rl_precision)
    rl_recall_mean = np.mean(rl_recall)

    precision_std = np.std(precision)
    recall_std = np.std(recall)
    rl_precision_std = np.std(rl_precision)
    rl_recall_std = np.std(rl_recall)

    # t-test (two tailed)
    statistic, pvalue = ttest_rel(original_score_list, adjusted_score_list)

    writer.flush()
    print("Scoring Completed")
    print("Original Image:")
    print(f"  Precision: Mean = {precision_mean:.4f}, Std = {precision_std:.4f}")
    print(f"  Recall: Mean = {recall_mean:.4f}, Std = {recall_std:.4f}")

    print("Adjusted Image:")
    print(f"  RL Precision: Mean = {rl_precision_mean:.4f}, Std = {rl_precision_std:.4f}")
    print(f"  RL Recall: Mean = {rl_recall_mean:.4f}, Std = {rl_recall_std:.4f}")

    print("TTest Result:")
    print(f"  statistic: = {statistic}, pvalue = {pvalue}")
