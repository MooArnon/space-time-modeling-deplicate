#--------#
# Import #
#------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.func  as F
import torch.nn.functional as F

#------------------------#
# Stacked Neuron network #
#------------------------------------------------------------------------#
class NNModel(nn.Module):
    """
    NNModel
    =======
    The simple linear layer with stacked number of layer.
    Each layer contains the relu activation.
    """
    
    def __init__(
            self, 
            input_size: int, 
            redundance: int = 4,
            num_layers: int = 5, 
            hidden_size: int = 1024
    ):
        """Initiate class

        Parameters
        ----------
        input_size : int :
            Size of input, might be window_size or number of features
        redundance: int :
            The reduction denominator of each layer
            Default is 4
        num_layers : int :
            Number of linear layers
            Default is 5
        hidden_size : int :
            Default is 1024
        """
        
        # Call inherited module
        super(NNModel, self).__init__()

        # Call linear
        self.linears = nn.ModuleList()

        # Create the linear layers
        for _ in range(num_layers):
            
            # Half reduction at each stacked layer
            hidden_size = hidden_size // redundance
            
            # Linear layer
            linear_layer = nn.Linear(input_size, hidden_size)
            
            # Append
            self.linears.append(linear_layer)
            
            # New input_size
            input_size = hidden_size

        # Output size
        self.fc_final = nn.Linear(hidden_size, 1)

    #------------------------------------------------------------------------#
    
    def forward(self, x):
        
        # Iterate over stacked linear layer
        for linear_layer in self.linears:
            
            # Generate output
            x = F.relu(linear_layer(x))

        # Output layer
        x = self.fc_final(x)
        
        return x

    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#