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
    """_summary_

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
    
    deep
    ----
    Perform the ordinary deep learning model, neuron network based
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
    """
    default_param = engine_dict[engine]["default_param"]

    new_kwargs = {
        key: default_param[key] if key not in kwargs.keys() else kwargs[key]
        for key in default_param.keys()
    }
    return engine_dict[engine]["engine"](**new_kwargs)
