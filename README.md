# space-time-modeling
I developed the `space_time_modeling` for the regression problems. <br>
You guys can perform the modeling by following the instruction at `run_local.py`, <br>
or, just use that code to run it.

PS. I always update the feature of this package. In the future, it can be run without the manual assignation, just use one function.

# How to
## Installation
Just paste the below script into the terminal or command line. <br>
`pip install git+https://github.com/MooArnon/space-time-modeling.git`
## Preprocessing
It can be used by import `get_preprocess_engine` from `space_time_modeling`
You need to specify the engine.
1. `series`: This is my first constructed engine. The engine will use target column to construct the x and y variable. The Kwags variable of this engine contains<br>
    - column : str
        Target column
    - mode : str, optional
        Mode of the source of data, 
        by default "csv"
    - diff : bool, optional
        If True, calculate diff and use it as features
        If False, Use the target column
    - window_size : int, optional
        The size of a input window, 
        by default 60 
```python
# Import preprocessing sub-package
from space_time_modeling import get_preprocess_engine 

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
```

## Modeling
After x and y was constructed. The modeling package will be used to perform the regression analysis of the data.
You need to specify the engine.
1. `deep`: This engine will use the deep learning model writ by torch to perform the analysis. This package is very flexible. You can use my build model to assign the architecture parameter, or, create your own model and pass it to the `regressor`
    - `architecture = nn`
        - input_size : int :
            Size of input, might be window_size or number of features
        - hidden_size : int :
            Number of nodes at the first layer.
            Default is 256
        - num_layers : int :
            Number of linear layers.
            Default is 5
        - redundance: int :
            The reduction denominator of each layer.
            Default is 4
    - `regressor = torch.nn.Module` if you want to customize your model. BUT, the model must receive `[batch_size, len(feature)]`.

```python
# Get engine
model_engine = get_model_engine(
    engine="deep",
    architecture = "nn",
    input_size = WINDOW_SIZE
)

# Train model
model_engine.modeling(
    x, 
    y, 
    result_name = "RNN",
    epochs=100,
    train_kwargs={"lr": 5e-5},
    test_ratio = 0.15
)
```
Arnon,
arnon.phongsiang@gmail.com

```project_directory/
├── space_time_modeling/
│   ├── __init__.py
│   |── preprocess.py
|   |   |──__init__.py
|   |   |──_base.py
|   |   └──...
|   |── modeling.py
|   |   |──__init__.py
|   |   |──_base.py
|   |   └──...
|   |── resources.py
|   |   |──__init__.py
|   |   └──...

├── tests/
|   |── test_base_series.py
|   |── test_modeling.py
|   |── test_series_preprocess
|   |   └──...

```
