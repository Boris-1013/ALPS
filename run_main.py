import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from scheduler import WarmUpLR, downLR
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

# 基本配置
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# 数据加载
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 预测任务
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# 模型定义
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=1, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default=768, help='LLM model dimension') # LLama7b:4096; GPT2-small:768; BERT-base:768

# 优化
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--input_size', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='/mnt/petrelfs/chengdawei/Boris/ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


for ii in range(args.itr):
    total_batch = 0  # 初始化 total_batch
    
    # 设置实验记录
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)
    
    # 数据加载
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    args.content = load_content(args)

    test_best_loss=float('inf')
    mae_best_loss=float('inf')
    # 初始化模型
    model = ALPS.Model(args)
    #model = ALPS_GPT2.Model(args)
    
    # 定义优化器、调度器
    optimizer = optim.Adam(model.parameters())
    scheduler = downLR(optimizer, (args.train_epochs - args.train_epochs / 2) * len(train_loader))
    warmup_scheduler = WarmUpLR(optimizer, args.train_epochs / 2 * len(train_loader))
    
    # 准备模型、数据加载器、优化器、调度器
    model, train_loader, vali_loader, test_loader, optimizer, scheduler, warmup_scheduler = accelerator.prepare(
        model, train_loader, vali_loader, test_loader, optimizer, scheduler, warmup_scheduler
    )

    # 创建检查点路径
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    train_steps = len(train_loader)

    start_time = time.time()
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    lrlist = np.zeros((args.train_epochs, 2))
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        accelerator.print('Epoch [{}/{}]'.format(epoch + 1, args.train_epochs))
        lrlist[epoch][0] = epoch
        if epoch >= args.train_epochs / 2:
            learn_rate = scheduler.get_lr()[0]
            accelerator.print("Learn_rate:%s" % learn_rate)
            lrlist[epoch][1] = learn_rate
        else:
            learn_rate = warmup_scheduler.get_lr()[0]
            lrlist[epoch][0] = learn_rate
            accelerator.print("Learn_rate:%s" % learn_rate)

        for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
            iter_count += 1
            optimizer.zero_grad()

            batch_x = batch_x.float() #Alibaba2022,两维
            batch_y = batch_y.float()
            # Decoder 输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

            # 混合精度训练
            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x, dec_inp)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]  # (B, L, D)
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]  # (B, L, D)
                    loss = nn.MSELoss()(outputs, batch_y)
                    total_batch += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x, dec_inp)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = nn.MSELoss()(outputs, batch_y)
                total_batch += 1
                accelerator.backward(loss)
                optimizer.step()

            train_loss.append(loss.item())
            # scheduler
            if epoch < args.train_epochs / 2:
                warmup_scheduler.step()
            else:
                scheduler.step()

        train_loss = np.average(train_loss)
        rmse_loss = np.sqrt(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, nn.MSELoss(), nn.L1Loss())
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, nn.MSELoss(), nn.L1Loss())
        
        if test_loss < test_best_loss:
            dev_best_loss = vali_loss
            test_best_loss = test_loss
        if test_mae_loss < mae_best_loss:
            mae_best_loss = test_mae_loss

        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Dev Loss: {2:>5.2},  Test Loss: {3:>5.2}, Time: {4}'
        accelerator.print(msg.format(total_batch, rmse_loss, vali_loss, test_loss, time_dif))
        accelerator.print('BEST SO FAR:')
        accelerator.print('Dev Best Loss:', dev_best_loss)
        accelerator.print('Test Best Loss:', test_best_loss)
        accelerator.print('MAE Best Loss:', mae_best_loss)
        
accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'  # 唯一的检查点保存路径
    del_files(path)  # 删除检查点文件
    accelerator.print('success delete checkpoints')
