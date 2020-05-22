# Project submission:
# Saurav Aryan
# Arthur Babey
# Stanislas Ducotterd

import torch
import math # This library is only used once for the constant PI

# Class containing linear module
class Linear():

    def __init__(self, dim_in, dim_out, initialize_He = True):
        """

        :param dim_in: Dimension of input to the layer
        :param dim_out: Dimension of the output from the layer
        :param initialize_He: Whether to initialize the parameters using He technique
        """
        # constant used for initializing weights
        # torch.manual_seed(54)
        eps = 1e-2

        # weights for the layer
        if initialize_He is True:
            self.w = torch.randn(dim_out, dim_in) * math.sqrt(2 / dim_in)  # He initialization technique
        else:
            self.w = torch.empty(dim_out, dim_in).normal_(0, eps)   # Simple weight initialization from normal distribution
        # bias for the layer
        # self.b = torch.empty(dim_out).normal_(0, eps)
        if initialize_He is True:
            self.b = torch.randn(dim_out) * math.sqrt(2/dim_in)  # He initialization technique
        else:
            self.b = torch.empty(dim_out).normal_(0, eps)
        # Corresponding gradients

        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

        # Temporary variable to store module's previous input for use in backward pass
        self.temp = torch.zeros(dim_in)

    def forward(self, x_in):
        """
        Function to give the output of the layer
        :param x_in: input sample, could be batch input as well for batch optimization
        :return: x_out: output of the layer
        """

        if x_in.dim() >1:
            x_out =  torch.mm(x_in, self.w.t()) + self.b         # To handle input of vector form
        else:
            x_out =  torch.mv(self.w,x_in) + self.b         # To handle input of tensor form (batch of samples)
        self.temp = x_in
        return x_out

    def gradient(self):         #gradient of output vs input

        return self.w.t()

    def backward(self, gradwrtoutput, x_in = None):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input, if not provided, taken from value stored in temp variable during forward pass
        :return: Gradient of loss wrt module's input
        """

        if x_in is None:            # If x_in is not provided, it's taken from the forward pass (as in test.py)
            x_in = self.temp
        if x_in.dim() == 1:         # Checking if the sample is individual or in batch
            self.dw += torch.ger(gradwrtoutput, x_in.t())       # Accumulate gradient wrt parameters
            self.db += gradwrtoutput
            dldx_in = torch.mv(self.w.t(), gradwrtoutput)  # Gradient of loss wrt the module's input

        else:                       # Same thing but for multiple training samples
            for id in range(x_in.shape[0]):
                self.dw += torch.ger(gradwrtoutput[id], x_in[id].t())  # Accumulate gradient wrt parameters
                self.db += gradwrtoutput[id]
            self.dw /= x_in.shape[0]
            self.db /= x_in.shape[0]
            dldx_in = torch.mv(self.w.t(), gradwrtoutput.mean(0))  # Gradient of loss wrt the module's input

        return dldx_in

    def param(self):
        """

        :return: paramlist: list of parameters of the module (w,b)
                gradlist: list of gradients corresponding to the parameters (dw, db)
        """
        paramlist = [self.w] + [self.b]
        gradlist = [self.dw] + [self.db]
        return paramlist, gradlist

    def grad_zero(self):
        """
        Function to set the gradients of parameters (dw, db) to zero. Called at the beginning of each epoch
        :return:
        """
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    __call__ = forward      # Makes the class object callable with forward method as call function.

# Class containing ReLU module
class relu():

    def __init__(self):
        self.temp = []        # Temporary variable to store module's previous input for use in backward pass


    def forward(self, x_in):
        """

        :param x_in: input to the module
        :return: output of the module
        """
        self.temp = x_in      # Temporary variable to store module's previous input for use in backward pass
        return torch.where(x_in> 0, x_in, torch.zeros_like(x_in))

    def gradient(self, x_in):
        """
        Function to return the gradient of the module's output wrt to the module's input
        :param x_in: input
        :return: gradient of output wrt input
        """
        return torch.where(x_in> 0, torch.ones_like(x_in), torch.zeros_like(x_in))

    def backward(self, gradwrtoutput, x_in = None):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input
        :return: Gradient of loss wrt module's input
        """

        if x_in == None:            # If x_in is not provided, it's taken from the forward pass
            x_in = self.temp

        dldx_in = torch.mul(gradwrtoutput, self.gradient(x_in))     #Compute gradient wrt input of module
        # print("RELU backward: ", x_in, gradwrtoutput, dldx_in)
        return dldx_in

    def param(self):
        """
        Module does not have any parameter so returns empty list
        :return:
        """
        return []

    def grad_zero(self):
        """
        Module does not have any parameters so does nothing
        :return:
        """
        pass


    __call__ = forward

# Class containing Tanh module
class Tanh():

    def __init__(self):
        self.temp = []        # Temporary variable to store module's previous input for use in backward pass


    def forward(self, x_in):
        """

        :param x_in: input to the module
        :return: output of the module
        """

        self.temp = x_in      # Temporary variable to store module's previous input for use in backward pass
        return (torch.exp(x_in) - torch.exp(-x_in)) / (torch.exp(x_in) + torch.exp(-x_in))

    def gradient(self, x_in):
        """
        Function to return the gradient of the module's output wrt to the module's input
        :param x_in: input
        :return: gradient of output wrt input
        """

        return 1 - torch.pow(self.forward(x_in), 2)

    def backward(self, gradwrtoutput, x_in = None):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input
        :return: Gradient of loss wrt module's input
        """

        if x_in == None:            # If x_in is not provided, it's taken from the forward pass
            x_in = self.temp

        dldx_in = torch.mul(gradwrtoutput, self.gradient(x_in))     #Compute gradient wrt input of module
        return dldx_in

    def param(self):
        """
        Module does not have any parameter so returns empty list
        :return:
        """

        return []

    def grad_zero(self):
        """
        Module does not have any parameters so does nothing
        :return:
        """
        pass


    __call__ = forward

# Class containing Sequential module
class Sequential():

    def __init__(self, *modules):
        """

        :param modules: list of layers and activation functions in order

        """
        self.layers = modules
        self.temp = []        # Temporary variable to store module's previous input for use in backward pass



    def forward(self, x_in):
        """
        function for forward pass through all layers in the sequential list
        :param x_in: input data
        :return: output processed data
        """
        # x_out = torch.zeros_like(x_in)

        for layer in self.layers:           #Call forward function of each layer in order
            x_out = layer.forward(x_in)
            # print("Forward pass Seq: ", layer, x_in, x_out)
            x_in = x_out                    # output of the layer is passed as input to the next layer
        self.temp = x_in
        return x_out

    __call__ = forward          # Designating forward function as call function to make the class object callable

    def backward(self, gradwrtoutput, x_in = None):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input, usually not required and taken from temp variiable
        :return: Gradient of loss wrt module's input
        """

        if x_in is None:            # If x_in is not provided, it's taken from the forward pass
            x_in = self.temp
        grad_in = gradwrtoutput     # Input here is usually the gradient from loss function for backpropagation
        count = len(self.layers)    # Number of layers in the module
        for i in range(0,count):    # Calling backward pass of layers in reverse order
            grad_out = self.layers[count-i-1].backward(grad_in) #count -i-1 to start from the outermost layer towards first layer
            grad_in = grad_out      # Pass the gradient from the current layer as input to the previous layer

    def param(self):
        """
        returns the list of parameters and gradients of each layer
        :return: gradlist, paramlist
        """
        paramlist = []
        gradlist = []

        for layer in self.layers:
            try:
                layer_param, layer_grad = layer.param()
                paramlist = paramlist + layer_param
                gradlist = gradlist + layer_grad
            except ValueError:
                continue
        return paramlist, gradlist

    def grad_zero(self):
        """
        Sets the gradient of parameters of each layer zero by calling grad_zero method of each layer
        :return:
        """
        for layer in self.layers:
            layer.grad_zero()

# Class containing loss function Mean Squared Error
class loss_MSE():
    def __init__(self):
        """
        This module does not have any parameters nor needs any temporary variable to store data
        """
        pass

    def forward(self, x_out, x_target):
        """

        :param x_out: Output from the model
        :param x_target: Target (test_target/train_target) corresponding to the sample
        :return: Mean Squared Error between the two
        """
        return torch.sum(torch.pow(x_out-x_target,2))

    def backward(self, x_out, x_target):
        """
        returns the gradient of loss function wrt predicted output of the model
        :param x_out: output of the model
        :param x_target: target of the corresponding sample
        :return: gradient of loss function
        """
        return 2*(x_out - x_target)

    def param(self):

        return []

    __call__ = forward

# Class for ADAM optimizer
class opt_adam():
    """
    Adam optimizer implementation
    Usage: opt_adam(model) inside the loop running epochs after backward pass

    """

    def __init__(self, model):
        """

        :param model: model whose parameters need to be optimized

        """
        self.beta_1 = 0.9   # Generally chosen value
        self.beta_2 = 0.999 # Generally chosen value
        self.step = 0.01    # Optimizer Parameter
        self.epsilon = 10e-3    # Optimizer Parameter

        self.m = []         # List of gradients first moment for each layer
        self.v = []         # List of gradients second moment for each layer

        self.paramlist, self.gradlist = model.param()       # get the list of parameters and its gradients from model
        self.nb_layers = len(self.paramlist)                # number of layers
        for layer_params in self.paramlist:
            self.m.append(torch.zeros_like(layer_params))
            self.v.append(torch.zeros_like(layer_params))

        self.iter = 1           # Counter of iterations should start at 1 (with 0, denominator of m_hat will vanish)


    def optimize_step(self, model):
        """

        :param model: model whose parameters need to be optimized

        """

        paramlist, gradlist = model.param()
        for i,(layer_params, layer_grads) in enumerate(zip(paramlist, gradlist)):
        # for i in range(self.nb_layers):                 # Running loops  over layers
            self.m[i] = self.beta_1* self.m[i] + (1 - self.beta_1) * layer_grads
            self.v[i] = self.beta_2* self.v[i] + (1 - self.beta_2) * torch.pow(layer_grads, 2)

            m_hat = self.m[i] / ( 1- self.beta_1**self.iter) + (1 - self.beta_1) * layer_grads/(1 - self.beta_1**self.iter)
            v_hat = self.v[i] / (1 - self.beta_2**self.iter)
            # if i == 1:
            #     print(m_hat, self.m[i], layer_grads)
            #     print(v_hat, self.v[i])

            # Update step
            layer_params -= self.step * m_hat/ (torch.sqrt(v_hat) + self.epsilon)

        self.iter += 1          # increase the counter of iterations

# Function to generate dataset described in the problem statement
def generate_data(n_sample):
    """

    :param n_sample: Number of training and testing samples required
    :return: train_input, test_input, train_target, test_target
    """

    dist = 1/ math.sqrt(2 * math.pi)

    train_input = torch.empty(n_sample, 2).uniform_(0, 1)
    test_input = torch.empty(n_sample, 2).uniform_(0, 1)
    train_target = torch.zeros(n_sample, 2)
    test_target = torch.zeros(n_sample,2)

    temp = (train_input - 0.5).norm(p=2, dim=1)

    train_target[temp > dist,0] = 1
    train_target[temp <= dist, 1] = 1
    # train_target = 1 - train_target

    temp = (test_input - 0.5).norm(p=2, dim=1)
    test_target[temp > dist, 0] = 1
    test_target[temp <= dist, 1] = 1
    # test_target = 1 - test_target

    return train_input, test_input, train_target, test_target







