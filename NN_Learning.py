"""
This script, provides general train and test methods for a neural network in Pytorch
"""
import torch


def train(dataloader, model, loss_fn, optimizer, device):
    """
    Neural Network Training method
    :param dataloader: A torch.utils.data.DataLoader object containing desired dataset
    :param model: A torch.nn.Module object, defining the model to be trained
    :param loss_fn: The loss function for training
    :param optimizer: An optimizer from torch.optim
    :param device: Training device, e.g. 'cpu' or 'gpu'
    :return: -
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, _) in enumerate(dataloader):
        x = x.to(device)
        x = x.float()

        # Compute prediction error
        network_output = model(x)
        loss = loss_fn(network_output, x)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        loss_checkpoint, current = loss.item(), batch * len(x)
        print(f"loss: {loss_checkpoint:>7f}  [{current:>5d}/{size:>5d}]")

        # return loss for plot


def test(dataloader, model, loss_fn, device):
    """
    Neural Network Testing method
    :param dataloader: A torch.utils.data.DataLoader object containing desired dataset
    :param model: The trained torch.nn.Module
    :param loss_fn: The loss function of the network
    :param device: Processing device, e.g. 'cpu' or 'gpu'
    :return: -
    """
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x = x.float()
            network_output = model(x)
            test_loss += loss_fn(network_output, x).item()
            # correct += (network_output.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    # return loss for plot
