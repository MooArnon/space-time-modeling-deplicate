import math

import torch

class BaseModeling:
    
    def __init__(self) -> None:
        pass
    
    #-------------#
    # Sample data #
    #------------------------------------------------------------------------#
    
    def sample_test_train(
            self,
            x: list[list[float]],
            y: list[list[float]],
            **Kwargs
    ) -> tuple[
        list[list[float]], list[list[float]], 
        list[list[float]], list[list[float]]
    ]:
        # Check length
        assert  len(x) == len(y), f"""
            Length of x and y is {len(x)}, {len(y)}; respectively """
        
        # Sample x
        x_train, x_test = self.sample(x, **Kwargs)
        
        # Sample y
        y_train, y_test = self.sample(y, **Kwargs)
        
        return x_train, y_train, x_test, y_test
        
    #------------------------------------------------------------------------#
    
    @staticmethod
    def sample(
            list: list[list[float]],
            test_ratio: float = 0.15,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Sample data in to the high and small one

        Parameters
        ----------
        list : list[list[float]]
            Target list
        test_ratio : float, optional
            Ratio of test data
            default 0.15

        Returns
        -------
        tuple(list[list[float]], list[list[float]])
            high, small
        """
        # Get the left partition
        left_partition_point = len(list) - math.ceil(len(list)*test_ratio)
        
        # Slice to get the high value
        high = list[:left_partition_point]
        
        # Slice to get the low value
        small = list[left_partition_point:]
        
        return high, small
        
    #-------#
    # Train #
    #------------------------------------------------------------------------#
    
    def train(self):
        """Need to be created for train model.
        Each engine need to return its model.
        If there are any fine tuning, create the separated 
        train_element method.
        """
        raise NotImplementedError("Child classes need to implement this fn")
    
    #------------------------------------------------------------------------#
    
    def train_element(
            self, 
            x: list[list[float]], 
            y: list[list[float]]
    ) -> torch.nn.Module:
        """Created for train the model in the element training.
        Be used in iterative search or fine tuning.
        """
        pass
    
    #------------------------------------------------------------------------#
    
    def predict(self, x: list[list[float]], model: torch.nn.Module) -> float:
        """Predict label, in this case the value from classification.
        """
        
        with torch.no_grad():
            
            model.eval()
            
            pred = model(x)
            
        return pred
        
#----------------------------------------------------------------------------#
