#--------#
# Import #
#----------------------------------------------------------------------------#
import os

import pandas as pd

from space_time_modeling import get_preprocess_engine
from space_time_modeling import DeepModeling
from space_time_modeling.resources.deep_model import NNModel

#--------------#
# Process data #
#----------------------------------------------------------------------------#
WINDOW_SIZE = 10

df = pd.read_csv(
    os.path.join("tests", "BTC-USD.csv")
)

prep = get_preprocess_engine(
    column="Open", 
    window_size=WINDOW_SIZE,
    diff=False,
)

x, y = prep.process(df=df)


#----------#
# Modeling #
#----------------------------------------------------------------------------#
model_nn = NNModel(input_size=WINDOW_SIZE, hidden_size=1024, num_layers=4, redundance=1)

engine = DeepModeling(model_nn)

engine.train(
    x, y, 
    train_kwargs={"lr": 5e-5, "epochs":500}, 
    preprocess_kwargs={"test_ratio": 0.25}
)
