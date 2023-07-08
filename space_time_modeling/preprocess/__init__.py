#--------#
# Import #
#----------------------------------------------------------------------------#
from space_time_modeling.preprocess._base import BasePreprocessing
from space_time_modeling.preprocess.series import SeriesPreprocess

#--------#
# Engine #
#----------------------------------------------------------------------------#

engine_dict ={
    "series": {
        "engine": SeriesPreprocess,
        "default_param": {
            "column": "price",
            "mode": "csv",
            "diff": False,
            "window_size": 45
        }
    }
}

#---------------#
# Call function #
#----------------------------------------------------------------------------#

def get_preprocess_engine(
        engine: str = "series", 
        **kwargs
) -> BasePreprocessing:
    """Be used for select the preprocessing engine

    Parameters
    ==========
    engine: str, optional
        The preprocessed engine, 
        by default "series"
    **kwargs:
        The parameter of each engine.
    
    Returns
    =======
    BasePreprocessing
    
    kwargs
    ======
    If engine `series` was selected
    - column : str
        Target column
    - mode : str, optional
        Mode of source of data, 
        by default "csv"
    - diff : bool, optional
        If True, calculate diff and use it as an features
        If False, Use the target column
    - window_size : int, optional
        The size of input window, 
        by default 60
        
    Raise
    =====
    ValueError
        If user select the wrong engine
    
    Example
    =======
    >>> import pandas as pd
    >>> from spacetime_modeling import get_preprocess_engine
    >>> prep = get_preprocess_engine(column="Open", diff=False,
    >>>     window_size=5
    >>> )
    >>> df = pd.read_csv("PATH", "TO", "CSV")
    >>> x, y = prep.process(df=df)
    """
    # Raise something if engine is not proper
    if engine not in engine_dict.keys():
        
        raise ValueError(
            f""" engine {engine} is not suitable. 
            You need to choose one from this list {list(engine_dict.keys())}.
            """
        )
        
    default_param = engine_dict[engine]["default_param"]

    # Combine the unassigned parameters
    new_kwargs = {
        key: default_param[key] if key not in kwargs.keys() else kwargs[key]
        for key in default_param.keys()
    }
    return engine_dict[engine]["engine"](**new_kwargs)
