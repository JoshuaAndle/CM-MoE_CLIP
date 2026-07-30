import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

# from .. import datasets, templates, utils
import torch.optim as optim
import torch
import torch.nn.functional as F
import os

#!# Note: This is something like 10M parameters per task, I feel like we could definitely reduce the overall size somehow
### Probably the best option is reducing the AlexNet output dimension to be smaller, which impacts all AutoEncoders
class AutoEncoder(nn.Module):
    """
    The class defines the autoencoder model which takes in the features from the last convolutional layer of the
    Alexnet model. The default value for the input_dims is 256*13*13.
    """

    def __init__(self, input_dims=256 * 13 * 13, code_dims=100):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid())


    # def encoder_criterion(self, outputs, inputs):
    #     loss = nn.MSELoss()
    #     return loss(outputs, inputs)



    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x


class Alexnet_FE(nn.Module):
    """
    Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
    and get the most related model whilst training a new task in a sequence
    """

    def __init__(self, alexnet_model):
        super(Alexnet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])
        self.fe_model.train = False

    def forward(self, x):
        return self.fe_model(x)




