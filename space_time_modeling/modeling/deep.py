import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from ._base import BaseModeling

class DeepModeling(BaseModeling):
    
    def __init__(self, classifier: object) -> None:
        
        """Initiate the DeepModeling class. 
        
        Parameters
        ----------
        classifier: object :
            The classifier object which can custom the architectures.
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
        
        # Get classifiers
        self.classifier = classifier
        
    #------------------------------------------------------------------------#
    
    def train(
            self, 
            x: list[list[float]], 
            y: list[list[float]], 
            preprocess_kwargs: dict = {"test_ratio": 0.1},
            train_kwargs: dict = {"lr": 3e-3},
    ) -> torch.nn.Module:
        """Train model, this can be adding the search function

        Parameters
        ----------
        x : list[list[float]]
            Input features
        y : list[list[float]]
            Input label
        preprocess_kwargs : _type_, optional
            Preprocessing keyword augments, 
            by default {"test_ratio": 0.1}
        train_kwargs : _type_, optional
            Training keyword augments , 
            by default {"lr": 3e-3}

        Returns
        -------
        torch.nn.Module
            Trained model
        """
        # Sample test and train data
        x_train, y_train, x_test, y_test = self.sample_test_train(
            x,
            y,
            **preprocess_kwargs
        )

        # Return the trained model
        return self.train_element(
            x_train, 
            y_train, 
            validation=(x_test, y_test),
            **train_kwargs
        )
    
    #------------------------------------------------------------------------#
    
    def train_element(
            self,
            x_train: list[list[float]], 
            y_train: list[list[float]], 
            validation: tuple[list[list[float]], list[list[float]]],
            lr: float = 0.003,
            epochs: int = 50,
            batch_size: int = 32,
    ) -> torch.nn.Module :
        """_summary_

        Parameters
        ----------
        x_train : list[list[float]]
            Train features
        y_train : list[list[float]]
            Train label
        validation : tuple[list[list[float]], list[list[float]]]
            Validation data
        lr : float, optional
            Learning rate, 
            by default 0.003
        epochs : int, optional
            Number of epochs, 
            by default 50
        batch_size : int, optional
            Batch size, 
            by default 32

        Returns
        -------
        torch.nn.Module
            Trained model
        """
        # Initialize model
        model: torch.nn.Module = self.classifier
        
        # Select loss fn
        criterion = nn.MSELoss()
        
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
            model,
            optimizer,
            criterion,
            batch_size,
            x_train,
            y_train,
            x_test,
            y_test,
    ) -> torch.nn.Module:
        """_summary_

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
        
        for batch_idx in range(num_batches):
        
            #----------------------------- Train ----------------------------#
            
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
        
        # Evaluate model
        with torch.no_grad():
            
            # Predict output
            pred = model(x_test)
            
            # Calculate loss
            val_loss = criterion(pred, y_test).item()
            
            # Calculate r2
            val_r2 = r2_score(y_test.tolist(), pred.tolist())
                
        # Print progress
        print(
            f"  |--- Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val r2: {val_r2:.4f} | "
        )
    
    #------------------------------------------------------------------------#
    
#------------------------------------------------------------------------#