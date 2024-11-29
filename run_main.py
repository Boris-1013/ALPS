import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from utils.scheduler import WarmUpLR, downLR
import time
from tqdm import tqdm
from datetime import timedelta
from models import ALPS, ALPS_GPT2
from data_provider.data_pre import data_provider
import random
import numpy as np
import os
import sys

from utils.tools import del_files, vali, load_content

parser = argparse.ArgumentParser(description='ALPS')


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loading
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--data', type=str, required='Alibaba2022')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 预测任务
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model defination
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=1, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension') # LLAMA:4096; GPT2:768

# 优化
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--eval_epochs', type=int, default=1, help='eval epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--input_size', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='/ALPS/ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


for ii in range(args.itr):
    total_batch = 0  # Initialize total batch counter
    
    # Set up experiment settings string
    setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff, ii)
    
    # Load data
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    args.content = load_content(args)


    dev_best_loss = float('inf')
    test_best_loss = float('inf')
    dev_mae_best_loss = float('inf')
    test_mae_best_loss = float('inf')

    # Initialize model
    model = ALPS.Model(args)
    #model = ALPS_GPT2.Model(args)
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters())
    scheduler = downLR(optimizer, (args.train_epochs - args.train_epochs / 2) * len(train_loader))
    warmup_scheduler = WarmUpLR(optimizer, args.train_epochs / 2 * len(train_loader))
    
    # Define optimizer and learning rate scheduler
    model, train_loader, vali_loader, test_loader, optimizer, scheduler, warmup_scheduler = accelerator.prepare(
        model, train_loader, vali_loader, test_loader, optimizer, scheduler, warmup_scheduler
    )

    # Create checkpoint path
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    train_steps = len(train_loader)

    start_time = time.time()
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    lrlist = np.zeros((args.train_epochs, 2))

    # Start training loop
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        accelerator.print('Epoch [{}/{}]'.format(epoch + 1, args.train_epochs))

        # Track learning rate during training
        lrlist[epoch][0] = epoch
        if epoch >= args.train_epochs / 2:
            learn_rate = scheduler.get_lr()[0]
            accelerator.print("Learn_rate:%s" % learn_rate)
            lrlist[epoch][1] = learn_rate
        else:
            learn_rate = warmup_scheduler.get_lr()[0]
            lrlist[epoch][0] = learn_rate
            accelerator.print("Learn_rate:%s" % learn_rate)

        # Train the model for each batch
        for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
            iter_count += 1
            optimizer.zero_grad()

            batch_x = batch_x.float() #Alibaba2022,两维
            batch_y = batch_y.float()
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

            # Mixed precision training (if enabled)
            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x, dec_inp)
                    f_dim = 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]  # (B, L, D)
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]  # (B, L, D)
                    loss = nn.MSELoss()(outputs, batch_y)
                    total_batch += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x, dec_inp)
                f_dim = 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = nn.MSELoss()(outputs, batch_y)
                total_batch += 1
                accelerator.backward(loss)
                optimizer.step()

            train_loss.append(loss.item())

            # Scheduler step
            if epoch < args.train_epochs / 2:
                warmup_scheduler.step()
            else:
                scheduler.step()

        # Average training loss and RMSE
        train_loss = np.average(train_loss)
        rmse_loss = np.sqrt(train_loss)

        # Validation and test
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, nn.MSELoss(), nn.L1Loss())
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, nn.MSELoss(), nn.L1Loss())
        

        # Save best test and MAE losses
        if vali_loss < dev_best_loss:
            dev_best_loss = vali_loss
            test_best_loss = test_loss
        if vali_mae_loss < dev_mae_best_loss:
            dev_mae_best_loss = vali_mae_loss
            mae_best_loss = test_mae_loss

        # Print progress
        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Dev Loss: {2:>5.2},  Test Loss: {3:>5.2}, Time: {4}'
        accelerator.print(msg.format(total_batch, rmse_loss, vali_loss, test_loss, time_dif))
        accelerator.print('BEST SO FAR:')
        accelerator.print('Dev Best Loss:', dev_best_loss)
        accelerator.print('Test Best Loss:', test_best_loss)
        accelerator.print('MAE Best Loss:', mae_best_loss)
        
accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'
    del_files(path)  # Delete checkpoint files
    accelerator.print('success delete checkpoints')
