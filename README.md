# space-time-modeling
I developed the `space_time_modeling` for the regression problems. <br>
You guys can perform the modeling by following the instruction at `run_local.py`, <br>
or, just use that code to run it.

PS. I always update the feature of this package. In the future, it can be run without the manual assignation, just use one function.


## Preprocessing
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
```python
# Simple NN model
model_nn = NNModel(
    input_size=WINDOW_SIZE, 
    hidden_size=1024, 
    num_layers=4, 
    redundance=1
)

# Get modeling engine
engine = DeepModeling(model_nn)

# Train it
model = engine.train(
    x, y, 
    train_kwargs={"lr": 5e-5, "epochs":500}, 
    preprocess_kwargs={"test_ratio": 0.25}
)
```
Arnon,
arnon.phongsiang@gmail.com

```project_directory/
├── space_time_modeling/
│   ├── __init__.py
│   |── preprocess.py
|   |   |──__init__.py
|   |   |──...
|   |── modeling.py
|   |   |──__init__.py
|   |   |──...
└── tests/
    └── test_modeling.py
    └── test_preprocess.py
```