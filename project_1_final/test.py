
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
from src.model import bigConvNet_3, Siamese
from src.helpers import training, nb_errors


EPOCHS = 100
LR = 0.1
BATCH_SIZE = 100
ALPHA = 0.9

def run():
    """
    Run the training
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = prolog.generate_pair_sets(1000)


    siamese = Siamese(nn.Sequential(nn.Linear(10, 1500), nn.ReLU(), nn.Linear(1500, 2)))

    print('Training start now: ')
    train_loss, test_loss, test_accuracy, best_accuracy = training(siamese, train_input, train_target, \
    train_classes, test_input, test_target, test_classes, epochs=EPOCHS, batch_size = BATCH_SIZE, lr = LR, alpha=ALPHA)

    print('Accuracy of our best Siamese Network model: ', best_accuracy)



if __name__ == '__main__':

    run()
