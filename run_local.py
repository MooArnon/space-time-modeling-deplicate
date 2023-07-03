#--------#
# Import #
#----------------------------------------------------------------------------#
import os

import pandas as pd

from space_time_modeling import SeriesPreprocess
from space_time_modeling import DeepModeling
from space_time_modeling.resources.deep_model import NNModel

#--------------#
# Process data #
#----------------------------------------------------------------------------#
WINDOW_SIZE = 3

df = pd.read_csv(
    os.path.join("tests", "BTC-USD.csv")
).head(500)

prep = SeriesPreprocess(column="Open", window_size=WINDOW_SIZE)

x, y = prep.process(df=df)


#----------#
# Modeling #
#----------------------------------------------------------------------------#
engine = DeepModeling(NNModel(input_size=WINDOW_SIZE, hidden_size=100))

engine.train(x, y)
