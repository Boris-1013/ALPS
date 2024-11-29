import os 
import numpy as np
import torch

# Data preprocessing function
def preprocess_dat(args):
    """
    Preprocess data and labels, and split them into training, validation, and test sets.
    
    Parameters:
        args: An object containing hyperparameters such as seq_len and pred_len.
    
    Returns:
        train_data: Training data containing sequences and their corresponding labels.
        val_data: Validation data containing sequences and their corresponding labels.
        test_data: Test data containing sequences and their corresponding labels.
    """
    path = os.getcwd()  # Get the current working directory
    data = np.load(path + '/results/alibaba2022-data.npy')  # Load data
    data = data.astype('float32')  # Convert data to float32 type
    label = np.load(path + '/results/alibaba2022-label.npy')  # Load labels
    label = label.astype('float32')  # Convert labels to float32 type

    # Split the dataset into train, validation, and test sets
    train_length = int(0.7 * data.shape[0])  # 70% for training
    val_length = int(0.8 * data.shape[0])  # 10% for validation, remaining 20% for testing

    # Split data
    data1 = data[:train_length]  # Training data
    data2 = data[train_length:val_length]  # Validation data
    data3 = data[val_length:]  # Test data

    # Split labels
    label1 = label[:train_length]  # Training labels
    label2 = label[train_length:val_length]  # Validation labels
    label3 = label[val_length:]  # Test labels

    # Construct training data
    train_data = []
    for x in range(data1.shape[0] - args.seq_len - args.pred_len):
        # Create samples: use seq_len sequence length to predict pred_len labels
        train_data.append((data1[x:x + args.seq_len], 
                           label1[x + args.seq_len - 1:x + args.seq_len + args.pred_len - 1]))

    # Construct validation data
    val_data = []
    for x in range(data2.shape[0] - args.seq_len - args.pred_len):
        val_data.append((data2[x:x + args.seq_len], 
                         label2[x + args.seq_len - 1:x + args.seq_len + args.pred_len - 1]))

    # Construct test data
    test_data = []
    for x in range(data3.shape[0] - args.seq_len - args.pred_len):
        test_data.append((data3[x:x + args.seq_len], 
                          label3[x + args.seq_len - 1:x + args.seq_len + args.pred_len - 1]))

    return train_data, val_data, test_data


# Data loader function
def data_provider(args, flag):
    """
    Returns the corresponding dataset and DataLoader based on the flag.
    
    Parameters:
        args: An object containing hyperparameters such as batch_size, num_workers, etc.
        flag: A string indicating the dataset type ('train', 'val', or 'test').
    
    Returns:
        The corresponding dataset (train_data, val_data, test_data) and DataLoader.
    """
    train_data, val_data, test_data = preprocess_dat(args)  # Preprocess the data
    if flag == 'train':
        # Create a DataLoader for the training set
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False  # Do not shuffle the data
        )
        return train_data, train_loader
    elif flag == 'val':
        # Create a DataLoader for the validation set
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False  # Do not shuffle the data
        )
        return val_data, val_loader
    else:
        # Create a DataLoader for the test set
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False  # Do not shuffle the data
        )
        return test_data, test_loader
