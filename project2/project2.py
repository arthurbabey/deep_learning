import torch
import math
import dlc_practical_prologue as prologue

class Linear():

    def __init__(self, dim_in, dim_out):

        # constant used for initializing weights
        eps = 1e-6

        # weights for the layer
        self.w = torch.empty(dim_out, dim_in).normal_(0, eps)

        # bias for the layer
        self.b = torch.empty(dim_out).normal_(0, eps)

        # Corresponding gradients

        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

        # Temporary variable to store module's previous input for use in backward pass
        self.temp = torch.zeros(dim_in)


    def forward(self, x_in):

        if x_in.dim() >1:
            x_out =  torch.mm(self.w,x_in) + self.b         # To handle input of vector form
        else:
            x_out =  torch.mv(self.w,x_in) + self.b         # To handle input of tensor form
        self.temp = x_in
        return x_out

    def gradient(self):         #gradient of output vs input

        return self.w.t()

    def backward(self, gradwrtoutput, x_in = None):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input
        :return: Gradient of loss wrt module's input
        """

        if x_in is None:            # If x_in is not provided, it's taken from the forward pass
            x_in = self.temp

        self.dw += torch.ger(gradwrtoutput, x_in.t())       # Accumulate gradient wrt parameters
        self.db += gradwrtoutput

        dldx_in = torch.mv(self.w.t(), gradwrtoutput)   # Gradient of loss wrt the module's input
        return dldx_in


    def param(self):
        paramlist = [self.w] + [self.b]
        gradlist = [self.dw] + [self.db]
        return paramlist, gradlist

    def grad_zero(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    __call__ = forward

class relu():

    def __init__(self):
        self.temp = []        # Temporary variable to store module's previous input for use in backward pass


    def forward(self, x_in):

        self.temp = x_in
        return torch.where(x_in> 0, x_in, torch.zeros_like(x_in))

    def gradient(self, x_in):

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
        return dldx_in

    def param(self):

        return []

    def grad_zero(self):
        pass


    __call__ = forward





class loss_MSE():
    def __init__(self):
        pass

    def forward(self, x_out, x_target):

        return torch.sum(torch.pow(x_out-x_target,2))

    def backward(self, x_out, x_target):

        return 2*(x_out - x_target)

    def param(self):

        return []

    __call__ = forward


class Sequential():

    def __init__(self, *modules):
        self.layers = modules
        self.temp = []        # Temporary variable to store module's previous input for use in backward pass



    def forward(self, x_in):
        """
        function for forward pass through all layers in the sequential list
        :param x_in: input data
        :return: output processed data
        """
        x_out = torch.zeros_like(x_in)

        for layer in self.layers:           #Call forward function of each layer
            x_out = layer.forward(x_in)
            x_in = x_out
        self.temp = x_in
        return x_out

    __call__ = forward

    def backward(self, gradwrtoutput, x_in = None):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input
        :return: Gradient of loss wrt module's input
        """

        if x_in is None:            # If x_in is not provided, it's taken from the forward pass
            x_in = self.temp
        grad_in = gradwrtoutput
        count = len(self.layers)
        for i in range(0,count):
            grad_out = self.layers[count-i-1].backward(grad_in)
            grad_in = grad_out

    def param(self):

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
        for layer in self.layers:
            layer.grad_zero()


if __name__ == '__main__':

    torch.set_grad_enabled(False)
    train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True, normalize=True)

    zeta = 0.90
    train_target = train_target * zeta
    test_target = test_target * zeta

    nb_train_samples = train_input.shape[0]
    nb_test_samples = test_input.shape[0]

    dim_in = train_input.shape[1]
    dim_out = train_target.shape[1]
    dim_hidden = 25

    lr = 0.01/nb_train_samples
    L1 = Linear(dim_in,dim_hidden)
    R1 = relu()
    L2 = Linear(dim_hidden,dim_out)
    R2 = relu()
    # L3 = Linear(dim_hidden,dim_out)
    # R3 = relu()
    loss_criterion = loss_MSE()

    model = Sequential(L1,R1, L2, R2)


    for e in range(0,1000):
        model.grad_zero()
        sum_loss = 0
        nb_train_errors = 0
        nb_test_errors = 0
        for idx in range(nb_train_samples):
            model_out = model(train_input[idx])
            if train_target[idx, model_out.argmax()] <0.5 :  # Checking if prediction is correct
                nb_train_errors += 1

            loss = loss_criterion(model_out, train_target[idx])
            loss_grad = loss_criterion.backward(model_out, train_target[idx])
            model.backward(loss_grad)                         # Backward step

            sum_loss = sum_loss + loss
        # Gradient Step
        paramlist, gradlist = model.param()
        for i, (param, param_grad) in enumerate(zip(paramlist, gradlist)):
            # print(param.max()/param_grad.max())
            param -= lr*param_grad

        for idx in range(nb_test_samples):
            model_out = model(test_input[idx])
            if test_target[idx, model_out.argmax()] < 0.5 :
                nb_test_errors += 1

        train_error = nb_train_errors/nb_train_samples * 100
        test_error = nb_test_errors/nb_test_samples * 100

        print("{:d} Loss= {:.02f}  Train error = {:.02f}%, Test error = {:.02f}%" .format(e, sum_loss,train_error, test_error))





