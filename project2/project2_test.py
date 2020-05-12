import torch
from project2 import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    torch.set_grad_enabled(False)
    train_input, test_input, train_target, test_target = generate_data(1000)

    zeta = 0.9
    train_target = train_target * zeta
    test_target = test_target * zeta

    nb_train_samples = train_input.shape[0]
    nb_test_samples = test_input.shape[0]

    dim_in = train_input.shape[1]
    dim_out = train_target.shape[1]
    dim_hidden = 25

    L1 = Linear(dim_in,dim_hidden)
    R1 = relu()
    L2 = Linear(dim_hidden,dim_hidden)
    R2 = relu()
    L3 = Linear(dim_hidden,dim_hidden)
    R3 = relu()
    L4 = Linear(dim_hidden,dim_hidden)
    R4 = relu()
    L5 = Linear(dim_hidden,dim_out)
    R5 = relu()

    # R1 = Tanh()
    # R2 = Tanh()
    # R3 = Tanh()
    # R4 = Tanh()
    # R5 = Tanh()

    loss_criterion = loss_MSE()

    model = Sequential(L1,R1, L2, R2, L3, R3, L4, R4, L5, R5)

    my_opt = opt_adam(model)            # Initialize the awesome Adam optimizer

    nb_epochs = 1000
    mini_batch_size = 1
    lr = 10e-8

    train_losses = []
    train_errors = []
    test_losses = []
    test_errors = []


    for e in range(0,nb_epochs):
        model.grad_zero()
        sum_loss = 0
        nb_train_errors = 0
        nb_test_errors = 0
        shuffled_ids = torch.randperm(nb_train_samples)
        train_input = train_input[shuffled_ids]     #Reshuffle input data at each epoch
        train_target = train_target[shuffled_ids]

        for batch in range(0, nb_train_samples, mini_batch_size):
            idx = range(batch, min(batch + mini_batch_size, nb_train_samples))
            model_out = model(train_input[idx])
            for vals in  train_target[idx, model_out.argmax(1)]:        # Checking if prediction is correct
                if vals <0.5 :
                    nb_train_errors += 1

            loss = loss_criterion(model_out, train_target[idx])
            loss_grad = loss_criterion.backward(model_out, train_target[idx])
            model.backward(loss_grad)                         # Backward step

            # print(loss, loss_grad)
            sum_loss = sum_loss + loss
            paramlist, gradlist = model.param()
            # Gradient Step

            for i, (param, param_grad) in enumerate(zip(paramlist, gradlist)):
                # print(param.max()/param_grad.max())
                param -= lr*param_grad
        # my_opt.optimize_step(model)

        for idx in range(nb_test_samples):
            model_out = model(test_input[idx])
            if test_target[idx, model_out.argmax()] < 0.5 :
                nb_test_errors += 1

        train_error = nb_train_errors/nb_train_samples * 100
        test_error = nb_test_errors/nb_test_samples * 100

        # print("{:d} Loss= {:.02f} Grad norm= ({:.02f}, {:.02f}, {:.02f}, {:.02f}, {:.02f}, {:.02f}) Train error = {:.02f}%, Test error = {:.02f}%"
        #       .format(e, sum_loss, gradlist[0].norm(), gradlist[1].norm(), gradlist[2].norm(), gradlist[3].norm(), gradlist[4].norm(), gradlist[-1].norm(), train_error, test_error))
        print("{:d} Loss= {:.02f}  Train error = {:.02f}%, Test error = {:.02f}%"
              .format(e, sum_loss, train_error, test_error))
        train_losses.append(sum_loss)
        train_errors.append(train_error)
        test_errors.append(test_error)



