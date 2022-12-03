from sklearn.datasets import make_classification,make_regression
# from urllib.request import Request, urlopen
import random
import pandas as pd
import numpy as np
from urllib.request import Request, urlopen
import random




def make_class(n_samples:int, n_features,n_informative, n_classes,shift:int=0.0,scale:int=None,random_state=None,shuffle=True):
    X, y = make_classification(n_samples=n_samples,n_features=n_features,n_informative=n_informative,n_classes=n_classes,
                                shift=shift,scale=scale,random_state=random_state,shuffle=shuffle)
    
    dataframe = pd.DataFrame(X)
    dataframe['label'] = y
    columnsnumber=random.sample(range(0,n_features-1),random.randint(0,n_features-1))
    
    dataframe[dataframe.columns[columnsnumber]]=dataframe[dataframe.columns[columnsnumber]].round()
    dataframe.columns = ['col_' + str(i) for i in dataframe.columns]
    
    return dataframe   




def make_regress(n_samples:int,n_features,n_informative,n_targets,bias=0.0,effective_rank=None,shuffle=True,random_state=None):
    X, y = make_regression(n_samples=n_samples,n_features=n_features,n_informative=n_informative,n_targets=n_targets,
                                bias=bias,effective_rank=effective_rank,random_state=random_state,shuffle=shuffle)
    
    dataframe1 = pd.DataFrame(X,columns=['col_' + str(i) for i in range(0,X.shape[1])])
    dataframe2=pd.DataFrame(y,columns=['target_' + str(i) for i in range(0,y.shape[1])])
    dataframe=pd.concat([dataframe1,dataframe2],1)
    columnsnumber=random.sample(range(0,n_features-1),random.randint(0,n_features-1))
    
    dataframe[dataframe.columns[columnsnumber]]=dataframe[dataframe.columns[columnsnumber]].round()
    # dataframe.columns = ['col_' + str(i) for i in dataframe.columns]
    
    return dataframe   





def make_cat(dataframe,column_num:int,replace:bool=True):

        dataframe_=dataframe.copy()


        url="https://www.mit.edu/~ecprice/wordlist.10000"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        web_byte = urlopen(req).read()

        webpage = web_byte.decode('utf-8')
        data = webpage.split("\n")
        data2=[]
        for item in data:
            if len(item)>4:
                data2.append(item)
        data3=random.sample(data2,int(random.random()*len(dataframe_)))
        cat_name=random.choice(data)
        dataframe_[cat_name]=[random.choice(data3) for _ in range(len(dataframe_))]
        column_to_move = dataframe_.pop(cat_name)
        dataframe_.insert(column_num, cat_name, column_to_move)
        if replace:
            dataframe_.drop(dataframe_.columns[column_num+1],1,inplace=True)
            

        return dataframe_




def inject_missing(dataframe,missingcolnumber:int,replace:bool=True,frac:float=.2):
    dataframe_main=dataframe.copy()
    dataframe_main.loc[dataframe.copy().sample(frac=frac).index, dataframe_main.columns[missingcolnumber]] = pd.np.nan
    return dataframe_main