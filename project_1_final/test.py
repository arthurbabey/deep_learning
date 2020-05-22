
"""
Run and train the best model, output best accuracy.
"""

import torch
import torch.optim as optim
import argparse
import os,sys
import src.dlc_practical_prologue as prolog

from torch import nn
from torch.nn import functional as F
from src.model import predictive_model
from src.helpers import training, nb_errors


EPOCHS = 1
LR = 0.1
BATCH_SIZE = 100

def run():
    """
    Run the training
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = prolog.generate_pair_sets(1000)

    #change the input data to fit in the training function
    train_input = train_input.view(-1, 14, 14).unsqueeze(1)
    test_input = test_input.view(-1, 14, 14).unsqueeze(1)
    train_classes = train_classes.view(2000)
    test_classes = test_classes.view(2000)
    test_target = test_target

    model = predictive_model(500, 0.5, 0.5) #initialize the predictive model
    comparison = nn.Sequential(nn.Linear(10, 1500), nn.ReLU(),nn.Linear(1500, 2))#initialize the comparison model
    comparisons = [comparison]
    print('Models are initialized !')



    print('Training start now:')

    _, _, _, best_accuracy, _ = training(model,comparisons,train_input, train_target, train_classes, test_input, test_target, test_classes, epochs=EPOCHS, batch_size = BATCH_SIZE, lr = LR)



    print('Training done !')
    print('Best accuracy of our best double model: ', best_accuracy.item())



if __name__ == '__main__':

    run()
