#%%
import json
import pandas as pd

clf = 'RF'
with open(f'{clf}_results_cv_bestparams.json', 'r') as f:
    combined = json.load(f)
