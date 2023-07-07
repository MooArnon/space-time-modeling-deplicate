import itertools

import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ._base import BaseModeling
from ..resources.deep_model.nn import NNModel

class DeepModeling(BaseModeling):
    
    default_architecture = {
        "nn": NNModel
    }
        
    def __init__(
            self, 
            regressor: object = None, 
            architecture: torch.nn.Module = None,
            **architecture_kwargs
    ) -> None:
        
        """Initiate the DeepModeling class. 
        
        Parameters
        ----------
        regressor: object :
            The regressor object which can custom the architectures.
            Need to be wrote in torch object. The input layer must
            receive the list[list[float]] which is 
            [batch_size,window_size]. Then, return the float value.
            IF NOT DEFINED, this function will call the default model
            from space_time_modeling/resources/deep_model which have
            2 type of model. You guys can use those code as a example.
        """
        # Chose the instance
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Use the default regressor
        if (regressor is None) & (architecture is None):
            
            self.regressor = self.default_architecture["nn"]()
        
        else:
            # set regressors
            self.regressor = regressor(**architecture_kwargs)
    
    #------------#  
    # Properties #    
    #------------------------------------------------------------------------#
    
    def set_regressor(self, regressor: object) -> torch.nn.Module:
        
        self.regressor = regressor
    
    #-------#
    # Train #
    #------------------------------------------------------------------------#
    
    def train(
            self, 
            x: list[list[float]], 
            y: list[list[float]], 
            test_ratio: float = 0.15,
            epochs: int = 100,
            train_kwargs: dict = None
    ) -> torch.nn.Module:
        """Train model, this can be adding the search function

        Parameters
        ----------
        x : list[list[float]]
            Input features
        y : list[list[float]]
            Input label
        test_ratio: float :
            The ratio of test that will be sampled.
        train_kwargs : _type_, optional
            Training keyword augments , 
            by default {"lr": 3e-3}

        Returns
        -------
        torch.nn.Module
            Trained model
        """
        # If engine kwargs was None,
        # Activate the default
        if train_kwargs is None:
            train_kwargs = {"lr": 3e-3}
        
        # Sample test and train data
        x_train, y_train, x_test, y_test = self.sample_test_train(
            x,
            y,
            test_ratio
        )

        # Return the trained model
        return self.train_element(
            x_train = x_train, 
            y_train = y_train, 
            validation=(x_test, y_test),
            epochs = epochs,
            **train_kwargs
        )
    
    #------------------------------------------------------------------------#
    
    def train_element(
            self,
            x_train: list[list[float]], 
            y_train: list[list[float]], 
            validation: tuple[list[list[float]], list[list[float]]],
            epochs: int = 100,
            lr: float = 0.003,
            batch_size: int = 32,
    ) -> torch.nn.Module :
        """Element training in case of fine tuning

        Parameters
        ----------
        x_train : list[list[float]]
            Train features
        y_train : list[list[float]]
            Train label
        validation : tuple[list[list[float]], list[list[float]]]
            Validation data with (x, y) format.
        epochs : int, optional
            Number of epochs, 
            by default 50
        lr : float, optional
            Learning rate, 
            by default 0.003
        batch_size : int, optional
            Batch size, 
            by default 32

        Returns
        -------
        torch.nn.Module
            Trained model
        """
        
        # Initialize model
        model: torch.nn.Module = self.regressor
        
        # Select loss fn
        criterion = nn.HuberLoss()
        
        # Select optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
        )
        
        # Force data in to the device
        x_train = torch.tensor(x_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)
        
        # Extract x and y test
        x_test = torch.tensor(validation[0])
        y_test = torch.tensor(validation[1])
        
        print(f"\n{model}")
        
        # Loop over epochs
        for epoch in range(epochs):
            
            print(f"Epoch: {epoch+1}/{epochs} | ")
            
            # Train in loop
            self.train_in_epoch(
                model,
                optimizer,
                criterion,
                batch_size,
                x_train,
                y_train,
                x_test,
                y_test,
            )
        
        return model
                
    #------------------------------------------------------------------------#
    
    def train_in_epoch(
            self,
            model: object,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.modules.loss,
            batch_size: int,
            x_train: list[list[float]],
            y_train: list[list[float]],
            x_test: list[list[float]],
            y_test: list[list[float]],
    ) -> torch.nn.Module:
        """Train in epoch

        Parameters
        ----------
        model : _type_
            Initialized model
        optimizer : _type_
            Optimizer
        criterion : _type_
            Loo function
        batch_size : _type_
            Batch size
        x_train : list[list[float]]
            Train features
        y_train : list[list[float]]
            Train label
        x_test : list[list[float]]
            Test features
        y_test : list[list[float]]
            Test label

        Returns
        -------
        torch.Module
            Trained model
        """
        # Calculate the number of batches
        num_batches = (x_train.shape[0] + batch_size - 1) // batch_size
        
        #------------------------------- Train ------------------------------#
        
        for batch_idx in range(num_batches):
            
            # Set model object
            model.train()
            optimizer.zero_grad()
            
            # generate the number of start index 
            # on each epoch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(x_train))
            
            # Slice input and label based on the calculated batch's idx
            input_batch = x_train[batch_start: batch_end]
            label_batch = y_train[batch_start: batch_end]
            
            # Get model output
            output = model(input_batch)
            
            # Calculate the loss
            loss = criterion(output, label_batch)
            
            # Back propagation
            loss.backward()
            optimizer.step()
            
            # Update loss value on each epoch
            train_loss = loss.item()
            
        #-------------------------- Evaluation ------------------------------#
        
        # Predict the data
        pred_train = self.predict(x_train, model, return_as="tensor")
        pred_val = self.predict(x_test, model, return_as="tensor")
        
        # Calculate loss
        val_loss = criterion(pred_val, y_test).item()
        
        # Calculate r2
        train_r2 = r2_score(y_train.tolist(), pred_train.tolist())
        val_r2 = r2_score(y_test.tolist(), pred_val.tolist())

        # Print progress
        print(
            f"  |--- Train Loss: {train_loss:.4f} | "
            f"Train r2: {train_r2:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val r2: {val_r2:.4f} | \n"
        )
    
    
    #---------------------------#
    # Evaluation and prediction #
    #------------------------------------------------------------------------#
    
    @staticmethod
    def predict(
            x: list[list[float]], 
            model: torch.nn.Module,
            return_as: str = "list"
    ) -> list[float]:
        """Use feature for predict the label

        Parameters
        ----------
        x : list[list[float]]
            Features
        model : torch.nn.Module
            Target model
        return_as: str :
            the type of input

        Returns
        -------
        list[float]
            the output
        """
        x = x.detach().clone()
        with torch.no_grad():

            prediction: torch.Tensor = model(x)
            
        # Format of data
        if return_as == "list":
            return prediction.tolist()

        elif return_as == "tensor":
            return prediction
    
    #------------------------------------------------------------------------#
    
    def plot_graph(
            self, 
            actual_value: list[float], 
            prediction_value: torch.Tensor
    ) -> None:
        """Generate the bar chart

        Parameters
        ----------
        actual_value : list[list[float]]
            The true value
        prediction_value : list[list[float]]
            the predicted value
        """
        # Convert tensor to list
        prediction_value = prediction_value.tolist()
        
        # Join list of list
        prediction_list = list(
            itertools.chain.from_iterable(prediction_value)
        )
        
        df = pd.DataFrame(
            {
                "prediction": prediction_list,
                "true_value": actual_value
            }
        )
        
        df = df.melt(
            value_vars=['y1', 'y2'], var_name='y', ignore_index=False
        )
        
        sns.lineplot(data=df, x=df.index, y='value', hue='y')
        
        plt.show()
        
    #-----------#
    # Utilities #
    #------------------------------------------------------------------------#
    
    def get_nn_model(self, architecture: str, **kwargs) -> torch.nn.Module:
        """Get the build in architecture

        Parameters
        ----------
        architecture : str
            if `nn`, use NNModel from space_time_modeling/resource/deep/nn.py

        Returns
        -------
        torch.Tensor
            The regressor module
        """
        
        if architecture == "nn":
            
            regressor = NNModel(**kwargs)
            
        return regressor
    
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
