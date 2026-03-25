import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from dataset import FusionDatasetSeq
from fusion_model import get_fusion_model
import args
from loss import final_ssim
import logging

from utils import set_seed
from warp import WarpingLayerBWFlow

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42
set_seed(seed)

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def train():

    logging.basicConfig(level=logging.INFO, filename="./data/log/new.log", filemode="a", format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = FusionDatasetSeq(args.train_vi_path, args.train_ir_path, args.train_flow_path, args.seq_path, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    print("train set num: ", len(train_dataset))

    model = get_fusion_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    os.makedirs(args.model_save_path, exist_ok=True)

    start_time = time.time()

    criterion = torch.nn.L1Loss()
    warp = WarpingLayerBWFlow().cuda()

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (vi_images, ir_images, flows) in enumerate(train_loader):

            vi_images, ir_images, flows = vi_images.to(device), ir_images.to(device), flows.to(device)
            vi_images, ir_images, flows = vi_images.squeeze(0), ir_images.squeeze(0), flows.squeeze(0)

            optimizer.zero_grad()
            vi_images_ycrcb = RGB2YCrCb(vi_images)
            vi_images_y = vi_images_ycrcb[:,:1]
            vi_images_y_tm1 = vi_images_y[:-1]
            vi_images_y_t = vi_images_y[1:]

            ir_images_tm1 = ir_images[:-1]
            ir_images_t = ir_images[1:]

            if torch.rand(1).item() < 0.75:
                vi_input = torch.cat([vi_images_y_t, vi_images_y_t], dim=1)
                ir_input = torch.cat([ir_images_t, ir_images_t], dim=1)
            else:
                vi_input = torch.cat([vi_images_y_tm1, vi_images_y_t], dim=1)
                ir_input = torch.cat([ir_images_tm1, ir_images_t], dim=1)

            fusion_images = model(vi_input, ir_input)

            ssim_loss_value = 1 - final_ssim(ir_images_t, vi_images_y_t, fusion_images)

            source_images_max = torch.max(vi_images_y_t, ir_images_t)
            l1_loss_value = F.l1_loss(source_images_max, fusion_images)

            fusion_total_loss =  ssim_loss_value + 0.9 * l1_loss_value

            flows = flows[1:]

            vi_images_y_t_left = warp(vi_images_y_t, flows)
            ir_images_t_left = warp(ir_images_t, flows)

            vi_images_y_t_left = vi_images_y_t_left.tile(1,2,1,1)
            ir_images_t_left = ir_images_t_left.tile(1,2,1,1)

            fusion_images_t_left = model(vi_images_y_t_left, ir_images_t_left)

            vi_images_y_t = vi_images_y_t.tile(1,2,1,1)
            ir_images_t = ir_images_t.tile(1,2,1,1)

            fusion_images_right = model(vi_images_y_t, ir_images_t)
            fusion_images_right_t = warp(fusion_images_right, flows)

            loss_t = criterion(fusion_images_t_left, fusion_images_right_t)

            total_loss = fusion_total_loss +  0.01 * loss_t

            total_loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 20 == 0:
                elapsed_time = time.time() - start_time
                total_batches = args.num_epochs * len(train_loader)
                completed_batches = epoch * len(train_loader) + batch_idx + 1
                remaining_batches = total_batches - completed_batches
                remaining_time = elapsed_time / completed_batches * remaining_batches

                print(f'Epoch [{epoch + 1}/{args.num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss.item():.6f}. Elapsed time: {format_time(elapsed_time)}, Remaining time: {format_time(remaining_time)}.')
                logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss.item():.6f}. Elapsed time: {format_time(elapsed_time)}, Remaining time: {format_time(remaining_time)}.")

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f'model_epoch{epoch + 1}.pth'))

        scheduler.step()

def format_time(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


if __name__ == "__main__":
    train()
