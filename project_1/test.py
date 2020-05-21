
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


EPOCHS = 10
LR = 0.1
BATCH_SIZE = 100
ALPHA = 0.5

def run(cuda = False):
    """
    Run the training
    """
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = prolog.generate_pair_sets(1000)

    siamese = Siamese(nn.Sequential(nn.Linear(10, 512), nn.ReLU(), nn.Linear(512, 512),\
    nn.ReLU(), nn.Linear(512, 2)))

    siamese = Siamese(nn.Sequential(nn.Linear(10, 512), nn.ReLU(), nn.Linear(512, 512),\
                                               nn.ReLU(), nn.Linear(512, 2)))
    if cuda:
        train_input = train_input.cuda()
        test_input = test_input.cuda()
        train_classes = train_classes.cuda()
        test_classes = test_classes.cuda()
        train_target = train_target.cuda()
        test_target = test_target.cuda()
        siamese = siamese.cuda()

    train_loss, test_loss, test_accuracy, best_accuracy = training(siamese, train_input, train_target, \
    train_classes, test_input, test_target, test_classes, epochs=EPOCHS, batch_size = BATCH_SIZE, lr = LR, alpha=ALPHA)

    print('Accuracy of our best Siamese Network model: ', best_accuracy)


def parse_args():
    """
    Parse command line flags.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', action='store_true', default=False, dest='cuda', help='Use GPU')

    results = parser.parse_args()

    return {'cuda': results.cuda}



if __name__ == '__main__':

    args = parse_args()
    run(args['cuda'])
