import torch
import math

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

    def forward(self, x_in):

        if x_in.dim() >1:
            x_out =  torch.mm(self.w,x_in) + self.b         # To handle input of vector form
        else:
            x_out =  torch.mv(self.w,x_in) + self.b         # To handle input of tensor form
        return x_out

    def gradient(self):         #gradient of output vs input

        return self.w.t()

    def backward(self, gradwrtoutput, x_in):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input
        :return: Gradient of loss wrt module's input
        """

        self.dw += torch.ger(gradwrtoutput, x_in.t())       # Accumulate gradient wrt parameters
        self.db += gradwrtoutput

        dldx_in = torch.mv(self.w.t(), gradwrtoutput)   # Gradient of loss wrt the module's input
        return dldx_in


    def param(self):
        paramlist = [self.w] + [self.b]
        gradlist = [self.dw] + [self.db]
        return paramlist, gradlist


    __call__ = forward

class relu():

    def __init__(self):
        pass

    def forward(self, x_in):

        return torch.where(x_in> 0, x_in, torch.zeros_like(x_in))

    def gradient(self, x_in):

        return torch.where(x_in> 0, torch.ones_like(x_in), torch.zeros_like(x_in))

    def backward(self, gradwrtoutput, x_in):
        """
        :param gradwrtoutput: Gradient of loss wrt module's output
        :param x_in: Module's input
        :return: Gradient of loss wrt module's input
        """
        dldx_in = torch.mul(gradwrtoutput, self.gradient(x_in))     #Compute gradient wrt input of module
        return dldx_in

    def param(self):

        return []



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




if __name__ == '__main__':
    L1 = Linear(4,2)
    R1 = relu()
    L2 = Linear(2,3)
    R2 = relu()
    L3 = Linear(3,4)
    loss = loss_MSE()

 





