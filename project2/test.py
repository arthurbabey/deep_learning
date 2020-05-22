# Project submission:
# Saurav Aryan
# Arthur Babey
# Stanislas Ducotterd


from project2 import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    torch.set_grad_enabled(False)       # Disable the awesome Auto-grad feature of Torch since we don't need it here
    train_input, test_input, train_target, test_target = generate_data(1000)    # Generate 1000 train and test samples

    zeta = 0.9                          # Rescale parameter for the train and test targets (as in practical 3)
    train_target = train_target * zeta
    test_target = test_target * zeta

    nb_train_samples = train_input.shape[0]     #Number of train samples
    nb_test_samples = test_input.shape[0]       # Number of test samples

    dim_in = train_input.shape[1]               # Input dimension of samples
    dim_out = train_target.shape[1]             # Output dimension of samples
    dim_hidden = 25                             # Number of units of hidden layers

    L1 = Linear(dim_in,dim_hidden)              # Input layer
    R1 = relu()                                 #Activation function
    L2 = Linear(dim_hidden,dim_hidden)          # Hidden layer 1
    R2 = relu()
    L3 = Linear(dim_hidden,dim_hidden)          # Hidden layer 2
    R3 = relu()
    L4 = Linear(dim_hidden,dim_hidden)          # Hidden layer 3
    R4 = relu()
    L5 = Linear(dim_hidden,dim_out)             # Output layer
    R5 = relu()

    # Activate the below five lines to use Tanh instead of RELU as activation function
    # R1 = Tanh()
    # R2 = Tanh()
    # R3 = Tanh()
    # R4 = Tanh()
    # R5 = Tanh()

    loss_criterion = loss_MSE()                 # Loss function

    model = Sequential(L1,R1, L2, R2, L3, R3, L4, R4, L5, R5)   # Constructing the sequential

    # my_opt = opt_adam(model)            # Activate to initialize the awesome Adam optimizer
                                            # Please disable batch SGD part in the loop for using this optimizer

    nb_epochs = 1000                        # Number of epochs
    mini_batch_size = 10                    # Mini batch size for batch SGD
    lr = 10e-3                              # Learning rate
    print("Minibatch size: ", mini_batch_size, " learning rate: ", lr)
    train_losses = []                       # List to append train losses at each epoch
    train_errors = []                       # List to append train errors at each epoch
    test_losses = []                        # List to append test losses at each epoch
    test_errors = []                        # List to append test errors at each epoch

    start_time = time.perf_counter()        # To check the time required for entire computation

    for e in range(0,nb_epochs):            # Run loops over epoch
        # model.grad_zero()
        sum_loss = 0
        sum_test_loss = 0
        nb_train_errors = 0
        nb_test_errors = 0
        shuffled_ids = torch.randperm(nb_train_samples)     # Shuffle training data for Stochastic Gradient descent
        train_input = train_input[shuffled_ids]     #Reshuffle input data at each epoch
        train_target = train_target[shuffled_ids]

        for batch in range(0, nb_train_samples, mini_batch_size):
            idx = range(batch, min(batch + mini_batch_size, nb_train_samples))
            model_out = model(train_input[idx])                         # Compute model's output (Forward pass)
            for vals in  train_target[idx, model_out.argmax(1)]:        # Checking if prediction is correct
                if vals <0.5 :
                    nb_train_errors += 1

            model.grad_zero()                               # Set the parameter gradients of model to zero
            loss = loss_criterion(model_out, train_target[idx]) # Calculate loss
            loss_grad = loss_criterion.backward(model_out, train_target[idx])   # Calculate loss gradient for backward pass
            model.backward(loss_grad)                         # Backward step

            # print(loss, loss_grad)
            sum_loss = sum_loss + loss                          # Accumulate losses
            paramlist, gradlist = model.param()                 # Get list of parameters and gradients of the model


            for i, (param, param_grad) in enumerate(zip(paramlist, gradlist)): # Gradient Step
                # print(param.max()/param_grad.max())
                param -= lr*param_grad

            # my_opt.optimize_step(model)                       # Activate this to use Adam's optimizer after disabling
                                                                # SGD optimizer above

        # Test samples
        for idx in range(nb_test_samples):                      # Computing test losses and test errors
            model_out = model(test_input[idx])
            loss = loss_criterion(model_out, test_target[idx])
            sum_test_loss  += loss
            if test_target[idx, model_out.argmax()] < 0.5 :
                nb_test_errors += 1

        train_error = nb_train_errors/nb_train_samples * 100        # Converting to percentage
        test_error = nb_test_errors/nb_test_samples * 100

        # print("{:d} Loss= {:.02f} Grad norm= ({:.02f}, {:.02f}, {:.02f}, {:.02f}, {:.02f}, {:.02f}) Train error = {:.02f}%, Test error = {:.02f}%"
        #       .format(e, sum_loss, gradlist[0].norm(), gradlist[1].norm(), gradlist[2].norm(), gradlist[3].norm(), gradlist[4].norm(), gradlist[-1].norm(), train_error, test_error))
        print("{:d} Train loss= {:.02f} Test loss = {:.02f}  Train error = {:.02f}%, Test error = {:.02f}%"
              .format(e, sum_loss, sum_test_loss, train_error, test_error))
        train_losses.append(sum_loss)
        train_errors.append(train_error)
        test_errors.append(test_error)
        test_losses.append(sum_test_loss)

    stop_time = time.perf_counter()
    duration = stop_time - start_time
    print('Time taken to run ', nb_epochs, ' epochs is ', duration)
    alpha = 0.7
    plt.plot(train_losses,  label = 'Train loss', alpha=alpha)
    plt.plot(test_losses,  label =  'Test loss', alpha=alpha)
    plt.legend()

    plt.figure()
    plt.plot(train_errors, label = 'Train error', alpha=alpha)
    plt.plot(test_errors, label = 'Test error', alpha=alpha)
    plt.legend()