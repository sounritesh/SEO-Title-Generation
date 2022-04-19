import torch
import torch.nn as nn
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device):
    '''
    Function to carry out training for all batches in an epoch

    Parameters:
    data_loader (torch.utils.data.Dataloader): data loader containing all training samples and labels.
    model (torch.nn.Module): classification model to be trained.
    optimizer (torch.optim.Optimizer): optimizer for fitting the model.
    device: device to load the model and tensors onto.

    Returns:
    epoch_loss (torch.Tensor): tensor of shape [1] reporting loss for the epoch.
    '''

    model.train()
    loss_tot = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
        ids = d["inputs"].to(device)
        labels = d["labels"].to(device)

        optimizer.zero_grad()

        output = model(ids, labels=labels)
        loss = output["loss"]

        loss.backward()
        loss_tot += loss.item()
        # print(loss.item())
        optimizer.step()

    epoch_loss =  loss_tot/len(data_loader)
    return epoch_loss


def eval_fn(data_loader, model, device):
    '''
    Function to carry out evaluation for all batches

    Parameters:
    data_loader (torch.utils.data.Dataloader): data loader containing all training samples and labels.
    model (torch.nn.Module): classification model to be trained.
    device: device to load the model and tensors onto.

    Returns:
    fin_outputs (torch.Tensor): output logits from the model.
    fin_targets (torch.Tensor): expected labels.
    '''

    model.eval()
    loss_tot = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
            ids = d["inputs"].to(device)
            labels = d["labels"].to(device)
            output = model(ids, labels=labels)
            loss = output["loss"]

            loss_tot += loss.item()

    epoch_loss =  loss_tot/len(data_loader)
    return epoch_loss