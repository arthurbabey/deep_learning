import torch
import torch.optim as optim

from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch import nn




def training(model, comparisons, train_input, train_target, train_classes, test_input, test_target,\
                       test_classes, epochs=50, batch_size=100, lr=0.1):
    """
    Function to train a double model.

    Model have to be a ConvNet with 2 layers
    Comparisons have to be a list of comparison model (10 dim input, 2 dim output)
    """

    torch.nn.init.xavier_uniform_(model.conv1.weight)
    torch.nn.init.xavier_uniform_(model.conv2.weight)
    torch.nn.init.xavier_uniform_(model.fc1.weight)
    torch.nn.init.xavier_uniform_(model.fc2.weight)

    for comparison in comparisons:
        for i in range(0, len(comparison), 2):
            torch.nn.init.xavier_uniform_(comparison[i].weight)

    optimizer_model = torch.optim.Adam(model.parameters())
    optimizer_comparisons = []
    for comparison in comparisons:
        optimizer_comparisons.append(torch.optim.Adam(comparison.parameters()))

    train_loss = torch.empty(70, len(comparisons), dtype=torch.float)
    test_loss = torch.empty(70, len(comparisons), dtype=torch.float)
    test_accuracy = torch.empty(70, len(comparisons), dtype=torch.float)
    best_accuracy = torch.empty(1, len(comparisons), dtype=torch.float)
    best_epoch = torch.empty(1, len(comparisons), dtype=torch.float)

    for i in range(100):
        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            criterion = torch.nn.CrossEntropyLoss()
            loss1 = criterion(output, train_classes.narrow(0, b, batch_size))
            optimizer_model.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer_model.step()

        mid_train = model(train_input).detach()
        mid_test = model(test_input).detach()
        mid_train_ = torch.zeros(int(mid_train.size(0) / 2), 10)
        mid_test_ = torch.zeros(int(mid_test.size(0) / 2), 10)

        for j in range(int(mid_train.size(0) / 2)):
                mid_train_[j,:] = mid_train[2*j,:] - mid_train[2*j + 1,:]
                mid_test_[j,:] = mid_test[2*j,:] - mid_test[2*j + 1,:]

        if i >= 30:
            for j in range(len(comparisons)):
                for k in range(epochs):
                    for b in range(0, mid_train_.size(0), batch_size):
                        output = comparisons[j](mid_train_.narrow(0, b, batch_size))
                        loss2 = criterion(output, train_target.narrow(0, b, batch_size))
                        optimizer_comparisons[j].zero_grad()
                        loss2.backward()
                        optimizer_comparisons[j].step()

                    output_train = comparisons[j](mid_train_)
                    output_test = comparisons[j](mid_test_)

                    train_loss[i-30][j] = criterion(output_train, train_target).item()
                    test_loss[i-30][j] = criterion(output_test, test_target).item()
                    accuracy = 1 - nb_errors(output_test, test_target) / 1000

                    if accuracy > best_accuracy[0][j]:
                        best_accuracy[0][j] = accuracy
                        best_epoch[0][j] = i+1
            test_accuracy[i-30][j] = accuracy

    return train_loss, test_loss, test_accuracy, best_accuracy, best_epoch



def train_siamese_model(siamese, train_input, train_target, train_classes, test_input, test_target,\
                       test_classes, epochs=100 , batch_size=100, lr=0.08, alpha=0.5):

    """
    Traning function for a siamese model
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
            output1 = siamese.forward1(train_input.narrow(0, b, batch_size)[:,0,:,:].unsqueeze(dim=1))
            output2 = siamese.forward1(train_input.narrow(0, b, batch_size)[:,1,:,:].unsqueeze(dim=1))
            criterion = torch.nn.CrossEntropyLoss()
            loss1 = criterion(output1, train_classes.narrow(0, b, batch_size)[:,0])
            loss2 = criterion(output2, train_classes.narrow(0, b, batch_size)[:,1])
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

    return train_loss, test_loss, test_accuracy, test_accuracy[-1]



def train_model(model, train_input, train_target, test_input, test_target,  epochs=500, batch_size=100, lr=0.1):
    """
    Training function for a training with 2 dimensional outputs.
    Model is assumed to be a ConvNet with 2 conv layers
    """

    torch.nn.init.xavier_uniform_(model.conv1.weight)
    torch.nn.init.xavier_uniform_(model.conv2.weight)

    optimizer = torch.optim.Adam(model.parameters())
    #scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_accuracy = 0
    best_epoch = 0

    for i in range(epochs):
        model.train()

        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        output_train = model(train_input)
        model.eval()
        output_test = model(test_input)
        train_loss.append(criterion(output_train, train_target).item())
        test_loss.append(criterion(output_test, test_target).item())
        accuracy = 1 - nb_errors(output_test, test_target) / 1000

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = i+1
        test_accuracy.append(accuracy)

        #if i%100 == 0:
        #    print('Epoch : ',i+1, '\t', 'test loss :', test_loss[-1], '\t', 'train loss', train_loss[-1])

    return train_loss, test_loss, test_accuracy, best_accuracy

def nb_errors(pred, truth):
    """
    Errors computation for 2 dimensional output training
    """

    pred_class = pred.argmax(1)
    return (pred_class - truth != 0).sum().item()


def nb_errors_10(pred, truth):
    """
    Errors computation for 10 dimensional output training
    """

    pred_class = pred.view(-1, 2, 10).argmax(2).argmax(1)
    return (pred_class - truth != 0).sum().item()


def train_model_10(model, train_input, train_classes, test_input, test_target, test_classes,\
                epochs=250, batch_size=100, lr=0.1):
    """
    Training function for 10 dimensional output.
    Model is assumed to be a convolutional network with 2 conv layers
    """

    torch.nn.init.xavier_uniform_(model.conv1.weight)
    torch.nn.init.xavier_uniform_(model.conv2.weight)

    optimizer = torch.optim.Adam(model.parameters())
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_accuracy = 0
    best_epoch = 0

    for i in range(epochs):
        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            criterion = torch.nn.CrossEntropyLoss()
            labels = train_classes.narrow(0, b, batch_size)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        output_train = model(train_input)
        output_test = model(test_input)
        train_loss.append(criterion(output_train, train_classes).item())
        test_loss.append(criterion(output_test, test_classes).item())
        accuracy = 1 - nb_errors_10(output_test, test_target) / 1000

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = i+1
        test_accuracy.append(accuracy)

        #if i%100 == 0:
            #print('Epoch : ',i+1, '\t', 'test loss :', test_loss[-1], '\t', 'train loss', train_loss[-1])

    return train_loss, test_loss, test_accuracy, best_accuracy, best_epoch