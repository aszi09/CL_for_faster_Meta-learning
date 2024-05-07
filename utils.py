import math
import torch
import torch.nn as nn

class VNet_(nn.Module):
    def __init__(self, input, hidden):
        super(VNet_, self).__init__()
        self.linear1 = nn.Linear(input, hidden)

    def forward(self, x):
        x = self.linear1(x)
        return torch.sigmoid(x)

def set_parameter(current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters