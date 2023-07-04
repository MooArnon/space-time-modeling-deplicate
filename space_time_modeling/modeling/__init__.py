#--------#
# Import #
#----------------------------------------------------------------------------#
from ..resources.deep_model import *
from .deep import DeepModeling
from ._base import BaseModeling

#--------#
# Engine #
#----------------------------------------------------------------------------#

engine_dict ={
    "deep": DeepModeling
}


#---------------#
# Call function #
#----------------------------------------------------------------------------#
def get_preprocess_engine(
        engine: str = "deep", 
        **kwargs
) -> BaseModeling:
    """Used to call the target modeling algorithm

    Parameters
    ==========
    engine: str, optional
        The modeling engine, 
        by default "deep"
    **kwargs:
        The parameter of each engine.

    Returns
    =======
    BaseModeling

    deep
    ----
    Perform the ordinary time series preprocess. Just sliding
    the interested column in data frame.
    classifier: object :
        Identify the classifier model. It must be torch.Module.
        At forward method must receive x: torch.tensor as an input.
        If you are struck with the model architecture, please apply the
        example at space_time_modeling/resource/deep_model/nn.py.
        nn: object :
            num_layers : int :
                Number of linear layers
                Default is 5
            hidden_size : int :
                Default is 1024
        
    """
    
    return engine_dict[engine](**kwargs)
