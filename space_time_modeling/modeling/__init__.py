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
    "deep": DeepModeling,
}

#---------------#
# Call function #
#----------------------------------------------------------------------------#
def get_model_engine(                                                   
        engine: str = "deep", 
        **kwargs
) -> BaseModeling:
    """Used to call the target modeling algorithm

    Parameters
    ==========
    engine: str, optional
        `deep` as a default. select the deep leaning algorithm.
    **kwargs:
        The parameter of each engine.

    Returns
    =======
    BaseModeling
    
    kwargs
    ======
    If engine `deep` was selected
    architecture: str :
        The architecture of deep model.
        - `nn` for stacked linear layer.
            - input_size : int :
                Size of input, might be window_size or number of 
                features
            - hidden_size : int :
                Number of node at the first layer.
                Default is 256
            - num_layers : int :
                Number of linear layers.
                Default is 5
            - redundance: int :
                The reduction denominator of each layer.
                Default is 4

    Raise
    =====
    ValueError
        If user select the wrong engine
        
    Examples
    ========
    >>> from space_time_modeling import
    >>> # Get engine
    >>> model_engine = get_model_engine(
    >>>     engine="deep", architecture="nn", input_size=3
    >>> )
    >>> # Train model
    >>> model_engine.modeling(
    >>>    x, 
    >>>    y, 
    >>>    result_name = "RNN",
    >>>    epochs=100,
    >>>    train_kwargs={"lr": 5e-5},
    >>>    test_ratio = 0.15
    >>> )
    """
    if engine not in engine_dict.keys():
        
        raise ValueError(
            f"""{engine} is not suitable. 
            You need to choose one from this list {list(engine_dict.keys())}.
            """
        )
        
    return engine_dict[engine](**kwargs)
