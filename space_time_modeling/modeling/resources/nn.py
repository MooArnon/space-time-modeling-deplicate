#--------#
# Import #
#------------------------------------------------------------------------#

import torch
import torch.nn as nn
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

#---------#
# N-BEATS #
#----------------------------------------------------------------------------#
# Main #
#------#
class NBEATS(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 256, 
        num_stacks: int = 2, 
        num_blocks: int = 2, 
        forecast_steps: int = 1
    ):
        """The N-BEATs model, torch model.
        
        Parameters
        ----------
        input_size : int :
            window size
        hidden_size : int :
            The hidden size, 
            by default 256
        num_stacks : int :
            The number of stacked nn layers, 
            by default 2
        num_blocks : int :
            The number of n-beats blocks, 
            by default 2
        forecast_steps : int :
            the step of forecast, 
            by default 1
        """
        super(NBEATS, self).__init__()
        
        self.blocks = nn.ModuleList(
            [NBEATSBlock(
                input_size, 
                hidden_size, 
                num_blocks, 
                forecast_steps
            ) for _ in range(num_stacks)]
        )
    
    #------------------------------------------------------------------------#
    
    def forward(self, x):
        stack_outputs = [block(x) for block in self.blocks]
        stack_outputs = torch.stack(stack_outputs, dim=1)

        return torch.mean(stack_outputs, dim=1)

#----------------------------------------------------------------------------#
# Element blocks #
#----------------#
class NBEATSBlock(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_blocks: int, 
        forecast_steps: int
    ):
        """The element block of n-beats

        Parameters
        ----------
        input_size : int
            Shape of input tensor,
            window_size in this case.
        hidden_size : int
            The size of hidden layer
        num_blocks : int
            The number of fully connected in block
        forecast_steps : int
            The number of forecasting steps
        """
        # Initialize the inherited class
        super(NBEATSBlock, self).__init__()
        
        # Trend sub-blocks
        self.trend_blocks = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for _ in range(num_blocks)]
        )
        
        # Seasonality block
        self.seasonality_blocks = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for _ in range(num_blocks)]
        )
        
        # Fully connected layers forecast
        self.fc1 = nn.Linear(hidden_size * num_blocks * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, forecast_steps)
    
    #------------------------------------------------------------------------#
    
    def forward(self, x: torch.tensor):
        """Forward method for NBEATSBlock class

        Parameters
        ----------
        x : torch.tensor
            The input tensor

        Returns
        -------
        torch.tensor
            result vector
        """
        # The list's result
        trend_outputs = []
        seasonality_outputs = []

        # Iterate over blocks, trend blocks and seasonality blocks
        for trend_block, seasonality_block in zip(
            self.trend_blocks, self.seasonality_blocks
        ):
            # Run model through each block
            # And store result in list
            trend_outputs.append(trend_block(x))
            seasonality_outputs.append(seasonality_block(x))

        # Concat sequence data
        trend_outputs = torch.stack(trend_outputs, dim=1)
        seasonality_outputs = torch.stack(seasonality_outputs, dim=1)

        # Concat trend_outputs and seasonality_outputs
        concatenated = torch.cat([trend_outputs, seasonality_outputs], dim=2)
        
        # Flatt data
        flattened = concatenated.view(concatenated.size(0), -1)

        # Activate relu function
        fc1_output = torch.relu(self.fc1(flattened))
        
        return self.fc2(fc1_output)
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
