# Casper
some random functions/classes which serve a utilitarian purpose in multiple projects
PS: Hierarchy of Classes will change as I add more utilities 

## Requirements
* pandas
* tensorboard
* pytorch

## Installing

```sh
git clone https://github.com/chahuja/pycasper.git
cd pycasper
python setup.py install
```

## ProtoTyping 
### Name (When finding unique names becomes a bigger task than the task itself)
* Based on a Namespace of arguments and its values, `Name` creates a unique name.
* If all arguments are too much for the length of the name, it is possible to provide a subset of those arguments
* Read more in the docfile of the class Name

### BookKeeper
* Logging routine for deep learning 

## Results
Visualize results stored by BookKeeper. 
I would recommend using jupyter lab / notebook to see all the results in a pretty format
```
import pandas as pd
from from pycasper.results import walkthroughResults
df, df_all = walkthroughResults('save/', args_subset=['exp', 'cpk', 'model', 'window_hop'],
                  res_subset=['train', 'dev', 'test'], val_key='dev')
print(df.sort_values(by='dev'))
```

PS: Ideally, it would work best if you use [BookKeeper](pycasper/BookKeeper.py) to store the results. But, if you store results in a json format, with 'train', 'test', 'dev' as dictionary keys storing lists of loss/metrics across number of epochs and args for the experiment as a python Dictionary, this function would work.

```sh
Results filename: <name>_res.json
Args filename: <name>_args.args
Both of them are json files
```
