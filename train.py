import argparse
from pathlib import Path
from tqdm import tqdm
import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd                                                 


import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from utils.general import increment_path

from ddpg_agent import DDPGAgent
from td3_agent import TD3
from dataset import OD_Dataset
from detect import yolo
from reinforcement import *


def record_training(save_dir, epochs, updateCnt):
    doc = "record.txt"
    with open(os.path.join(save_dir, doc), 'w') as f:
        f.write(str(epochs) + " " + str(updateCnt))


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Resident memory (RSS): {mem_info.rss / 1024**2:.2f} MB")
    print(f"Virtual memory (VMS): {mem_info.vms / 1024**2:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='coco')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--buffer-size', type=int, default=4000, help='buffer size')
    parser.add_argument('--max-timesteps', default=1000, type=int)
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--plot', action='store_true', help='plot?')
    parser.add_argument('--split_dataset', default=20000, type=int, help='split dataset')
    opt = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Result directary
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)  # increment run
    tb_writer = SummaryWriter(save_dir + "/metrics")
    pd_data = pd.DataFrame(columns=['reward', 'score_gap'])

    # Create dqn model & YOLOv7
    action_dim = 4
    agent = TD3(action_dim, save_dir=save_dir, buffer_size=opt.buffer_size, batch_size=opt.batch_size)
    init_epoch, update_cnt = 0, 0

    # yolov7 model
    yolo_model = yolo()

    # Trainloader
    trainDataset = OD_Dataset(opt.dataset_path, mode='train')
    subset_indices = random.sample(range(len(trainDataset)), opt.split_dataset)
    partial_dataset = Subset(trainDataset, subset_indices)
    trainDataloader = DataLoader(partial_dataset, batch_size=1, shuffle=True)

    #---------------------------------------#
    #   Start training
    #---------------------------------------#
    for epoch in range(init_epoch, opt.epochs):
        '''
            Random split dataset
        '''
        data_list = []
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

            for step in range(opt.steps):
                '''
                    action : [contrast, saturation, brightness]
                '''
                # Origin Image
                precision_origin, recall_origin = yolo_model.detectImg(img, target)
                Origin_score = get_score(precision_origin, recall_origin)

                # With RL Score
                if (len(agent.memory) > opt.max_timesteps):
                    action = (agent.select_action(img.unsqueeze(dim=0))[0] + np.random.normal(0, 1*0.1, size=action_dim)).clip(-1, 1)
                else:
                    action = np.random.uniform(-1, 1, size=action_dim)
                trans_action = transform_action(action, 0.5, 1.5)
                adjust_img = modify_image(img, *trans_action)
                precision_score_RL, recall_distortion_RL = yolo_model.detectImg(adjust_img, target)
                RL_score = get_score(precision_score_RL, recall_distortion_RL)

                reward = get_reward(RL_score, Origin_score)
                
                # Push experient to memory
                state = img
                next_state = adjust_img
                agent.add(state, action, reward, next_state)
                
                # Move to the next state
                state = next_state

                # Network train & update
                if (len(agent.memory) > opt.max_timesteps):
                    agent.train()

                # Print & Record
                pbar.set_description((f"Epoch [{epoch+1}/{opt.epochs}]"))

                # Writer
                tb_writer.add_scalar('reward', reward, update_cnt)
                tb_writer.add_scalar('score gap', RL_score-Origin_score, update_cnt)
                tb_writer.add_scalars('action/original_action', {'bright': action[0], 'saturation': action[1], 'contrast': action[2], 'sharpness': action[3]}, update_cnt)
                tb_writer.add_scalars('action/adjusted_action', {'bright': trans_action[0], 'saturation': trans_action[1], 'contrast': trans_action[2], 'sharpness': trans_action[3]}, update_cnt)
                data_list.append([reward, RL_score-Origin_score])
                
                update_cnt += 1 

                if opt.plot:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(TF.to_pil_image(img))
                    axs[0].set_title('origin image')   
                    axs[1].imshow(TF.to_pil_image(adjust_img))
                    axs[1].set_title('adjust image')      
                    plt.savefig('image.jpg')

        # end batch -------------------------------------------------------------
        tb_writer.flush()
        agent.save()    # Save model
        record_training(save_dir, epoch+1, update_cnt)    # Save training record

        df_extended = pd.DataFrame(data_list, columns=['reward', 'score_gap'])
        pd_data = pd.concat([pd_data, df_extended], ignore_index=True)
        pd_data.to_csv(os.path.join(save_dir, "metric.csv"), index=False)    # save metric

    # end epoch ---------------------------------------------------------

    # End training ---------------------------------------------------------
    print("End Training\n")