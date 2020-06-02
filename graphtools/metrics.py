import numpy as np
import pandas as pd
#%%
def f_score(data, feature, label):
    f1 = data.loc[data.loc[:,label]==0,feature]
    f2 = data.loc[data.loc[:,label]==1,feature]
    n1 = len(f1)
    n2 = len(f2)
    x1 = f1.mean()
    x2 = f2.mean()
    xw = data.loc[:,feature].mean()
    num = (x1-xw)**2+(x2-xw)**2
    s1 = 0
    for i in range(n1):
        s1+= (f1.iloc[i] - x1)**2
    s1=s1/(n1-1)
    s2 = 0
    for i in range(n2):
        s2+= (f2.iloc[i] - x2)**2
    s2=s2/(n2-1)
    deno = s1+s2
    if deno!=0:
        print(num, deno)
        return num/deno
    else:
        return 0

def fscore(data, class_col='class'):
    """ Compute the F-score for all columns in a DataFrame
    """
    grouped = data.groupby(by=class_col)
    means = data.mean()
    g_means = grouped.mean()
    g_vars = grouped.var()

    numerator = np.sum((g_means - means) ** 2, axis=0)
    denominator = np.sum(g_vars, axis=0)
    if sum(denominator)!=0:
        return round(numerator/denominator, 3)
    else:
        return np.zeros(numerator.shape)