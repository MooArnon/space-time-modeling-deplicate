#--------#
# Import #
#----------------------------------------------------------------------------#
import os

import pandas as pd

from space_time_modeling import get_preprocess_engine
from space_time_modeling import get_model_engine
from space_time_modeling.resources.deep_model import NNModel

#--------------#
# Process data #
#----------------------------------------------------------------------------#
WINDOW_SIZE = 10

# Read data
df = pd.read_csv(
    os.path.join("tests", "BTC-USD.csv")
)

# Get preprocessing engine
prep = get_preprocess_engine(
    column="Open", 
    window_size=WINDOW_SIZE,
    diff=False,
)

# Calculate x and y
x, y = prep.process(df=df)


#----------#
# Modeling #
#----------------------------------------------------------------------------#
# Simple NN model
model_nn = NNModel(
    input_size=WINDOW_SIZE,
    hidden_size=1024,
    num_layers=4,
    redundance=1
)

model_engine = get_model_engine(
    engine="deep",
    architecture = "nn",
    input_size = WINDOW_SIZE
)

# Train it
model_engine.modeling(
    x, 
    y, 
    result_name = "RNN",
    epochs=100,
    train_kwargs={"lr": 5e-5},
    test_ratio = 0.15
)

