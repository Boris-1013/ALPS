import os 
import numpy as np
import torch

def preprocess_dat(args):
    path=os.getcwd()
    data=np.load(path+'/results/alibaba2022-data.npy')
    data=data.astype('float32')
    label=np.load(path+'/results/alibaba2022-label.npy')
    label=label.astype('float32')
    train_length=int(0.7*data.shape[0])
    val_length=int(0.8*data.shape[0])
    
    #数据的划分
    data1=data[:train_length]
    data2=data[train_length:val_length]
    data3=data[val_length:]
    #标签的划分
    label1=label[:train_length]
    label2=label[train_length:val_length]
    label3=label[val_length:]
    train_data=[]
    for x in range(data1.shape[0]-args.seq_len-args.pred_len):
        train_data.append((data1[x:x+args.seq_len],label1[x+args.seq_len-1:x+args.seq_len+args.pred_len-1]))
    val_data=[]
    for x in range(data2.shape[0]-args.seq_len-args.pred_len):
        val_data.append((data2[x:x+args.seq_len],label2[x+args.seq_len-1:x+args.seq_len+args.pred_len-1]))
    test_data=[]
    for x in range(data3.shape[0]-args.seq_len-args.pred_len):
        test_data.append((data3[x:x+args.seq_len],label3[x+args.seq_len-1:x+args.seq_len+args.pred_len-1]))
    return train_data, val_data, test_data


def data_provider(args, flag):
    train_data, val_data, test_data=preprocess_dat(args)
    if flag=='train':
        train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=args.batch_size,num_workers=args.num_workers,
                                        shuffle=False)#在训练过程中，batch是指一次处理的多个(X, y)对的集合
        return train_data, train_loader
    elif flag=='val':
        val_loader=torch.utils.data.DataLoader(dataset=val_data,batch_size=args.batch_size,num_workers=args.num_workers,
                                            shuffle=False)
        return val_data, val_loader
    else:
        test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=args.batch_size,num_workers=args.num_workers,
                                            shuffle=False)
        return test_data,test_loader
