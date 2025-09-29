import torch
import torch.nn as nn

class Layer1(nn.Module):
    def __init__(self):
        super(Layer1, self).__init__()

    def forward(self, t):
        out = t**2
        return out
    
class Layer2(nn.Module):
    def __init__(self):
        super(Layer2, self).__init__()
        self.param = nn.Parameter(torch.tensor(3,dtype=torch.float32),requires_grad=True) 

    def forward(self, t):
        out = (self.param * t)**3
        return out.sum()
    
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layer1 = Layer1()
        self.layer2 = Layer2()

        
        self.layer1.register_forward_hook(self.save_activations)
        self.layer1.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out

    def forward(self, t):
        out = self.layer1(t)
        out = self.layer2(out)
        return out
    
model = model()  # Use GPU if available

t = torch.tensor([2,4], dtype=torch.float32, requires_grad=True)
output = model(t)

model.zero_grad()
output.backward()


activations = model.activations
gradients = model.gradients 
model.layer2.param.grad