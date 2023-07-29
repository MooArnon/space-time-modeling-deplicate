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
            hidden_size: int = 256,
            num_layers: int = 3, 
            redundance: int = 1,
    ) -> None:
        """Initiate class

        Parameters
        ----------
        input_size : int :
            Size of input, might be window_size or number of features
        hidden_size : int :
            Number of node at the first layer.
            Default is 256
        num_layers : int :
            Number of linear layers.
            Default is 5
        redundance: int :
            The reduction denominator of each layer.
            Default is 4
        """
        
        # Call inherited module
        super(NNModel, self).__init__()

        # Call linear
        self.linears = nn.ModuleList()
        
        self.input_size = input_size

        # Create the linear layers
        for _ in range(num_layers):
            
            # Half reduction at each stacked layer
            hidden_size = hidden_size // redundance
            
            # Linear layer
            linear_layer = nn.Linear(self.input_size, hidden_size)
            
            # Append
            self.linears.append(linear_layer)
            
            # New input_size
            self.input_size = hidden_size

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


#---------------------------#
# Stacked Recurrent network #
#------------------------------------------------------------------------#
class LSTMModel(nn.Module):
    def __init__(
            self,
            input_size: int, 
            hidden_size: int= 256, 
            num_layers: int= 2
    ):
        super(LSTMModel, self).__init__()
        
        self.rnn = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True, 
            num_layers = num_layers,
        )
        self.fc = nn.Linear(hidden_size, 1)

    #------------------------------------------------------------------------#
    
    def forward(self, x):
        
        # x = torch.unsqueeze(x, dim=2)
        
        out, _ = self.rnn(x)
        
        # out = torch.mean(out, dim=-1)

        out = self.fc(out)
        
        return out
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#