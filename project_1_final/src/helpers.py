import torch
import torch.optim as optim

from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch import nn




import torch
import torch.optim as optim

from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch import nn

def training(siamese, train_input, train_target, train_classes, test_input, test_target,\
                       test_classes, epochs=10 , batch_size=100, lr=0.08, alpha=0.5):
    """
    Function to train the whole siamese network
    """

    torch.nn.init.xavier_uniform_(siamese.model.conv1.weight)
    torch.nn.init.xavier_uniform_(siamese.model.conv2.weight)
    torch.nn.init.xavier_uniform_(siamese.model.fc1.weight)
    torch.nn.init.xavier_uniform_(siamese.model.fc2.weight)

    for i in range(0, len(siamese.comparison), 2):
        torch.nn.init.xavier_uniform_(siamese.comparison[i].weight)

    optimizer = torch.optim.Adam(siamese.parameters())
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_accuracy = 0
    best_epoch = 0

    for i in range(epochs):
        for b in range(0, train_input.size(0), batch_size):
            "We first predict the digit of each images"
            output1 = siamese.forward1(train_input.narrow(0, b, batch_size)[:,0,:,:].unsqueeze(dim=1))
            output2 = siamese.forward1(train_input.narrow(0, b, batch_size)[:,1,:,:].unsqueeze(dim=1))
            criterion = torch.nn.CrossEntropyLoss()
            loss1 = criterion(output1, train_classes.narrow(0, b, batch_size)[:,0])
            loss2 = criterion(output2, train_classes.narrow(0, b, batch_size)[:,1])
            "And then we predict the target"
            output3 = siamese.forward2(output1, output2)
            loss3 = criterion(output3, train_target.narrow(0, b, batch_size))
            loss = alpha*(loss1 + loss2) + (1 - alpha)*loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mid_train1 = siamese.forward1(train_input[:,0,:,:].unsqueeze(dim=1))
        train_loss1 = criterion(mid_train1, train_classes[:,0])
        mid_train2 = siamese.forward1(train_input[:,1,:,:].unsqueeze(dim=1))
        train_loss2 = criterion(mid_train2, train_classes[:,1])
        output_train = siamese.forward2(mid_train1, mid_train2)
        train_loss3 = criterion(output_train, train_target)

        mid_test1 = siamese.forward1(test_input[:,0,:,:].unsqueeze(dim=1))
        test_loss1 = criterion(mid_test1, test_classes[:,0])
        mid_test2 = siamese.forward1(test_input[:,1,:,:].unsqueeze(dim=1))
        test_loss2 = criterion(mid_test2, train_classes[:,1])
        output_test = siamese.forward2(mid_test1, mid_test2)
        test_loss3 = criterion(output_test, test_target)

        train_loss.append(((alpha*(train_loss1 + train_loss2) + alpha*train_loss3).item()))
        test_loss.append(((alpha*(test_loss1 + test_loss2) + alpha*test_loss3).item()))
        accuracy = 1 - nb_errors(output_test, test_target) / 1000

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = i+1
        test_accuracy.append(accuracy)

    return train_loss, test_loss, test_accuracy, best_accuracy




def nb_errors(pred, truth):
    """
    Compute the number of errors
    """

    pred_class = pred.argmax(1)
    return (pred_class - truth != 0).sum().item()


