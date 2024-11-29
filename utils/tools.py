import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

plt.switch_backend('agg')  # Set the matplotlib backend to 'agg' for saving plots without displaying them

def del_files(dir_path):
    #Deletes a directory and all its contents.

    shutil.rmtree(dir_path)  # Removes a directory and its content recursively


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    #Validates the model on the validation dataset.

    total_loss = []  
    total_mae_loss = [] 
    model.eval() 
    test_flag = True  

    with torch.no_grad():  # No gradient calculation during validation
        # Iterate through the validation data loader
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float() 
            batch_y = batch_y.float()

            # Create decoder input: concatenate past labels and zeros for future prediction
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # Perform forward pass through the model (encoder-decoder architecture)
            outputs = model(batch_x, dec_inp, test_flag)

            # Select relevant dimensions for the output and target
            f_dim = 0
            outputs = outputs[:, -args.pred_len:, f_dim:] 
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device) 

            pred = outputs.detach()  # Detach the predictions from the computation graph (no gradients)
            true = batch_y.detach()  # Detach the true labels from the computation graph

            # Calculate loss using the criterion
            loss = criterion(pred, true)

            # Calculate MAE (Mean Absolute Error)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())  # Store the loss for averaging
            total_mae_loss.append(mae_loss.item())  # Store the MAE for averaging

    # Calculate average RMSE and MAE over the entire validation set
    total_loss = np.average(total_loss)
    rmse_loss = np.sqrt(total_loss)  # RMSE is the square root of the average loss
    total_mae_loss = np.average(total_mae_loss)  # Average MAE

    model.train()  # Set the model back to training mode
    return rmse_loss, total_mae_loss  # Return the final RMSE and MAE values


def load_content(args):
    #Loads the prompt content based on the dataset.

    file = args.data  # Use the dataset name directly

    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()  # Read the content from the corresponding prompt file

    return content  # Return the content of the prompt
