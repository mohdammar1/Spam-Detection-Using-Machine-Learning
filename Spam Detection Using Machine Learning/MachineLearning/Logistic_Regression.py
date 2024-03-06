import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
process=pd.read_csv('MachineLearning/finaldata')
x=process['Message']
y=process['Category']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=3)
feature_extraction=TfidfVectorizer(min_df=1, stop_words='english' , lowercase=True)
x_train_feature=feature_extraction.fit_transform(x_train)
x_test_feature=feature_extraction.transform(x_test)
y_train=y_train.astype('int')
y_test=y_test.astype('int')
spam='Spam Detected'
nospam='No Spam Detected'
class Ammar:
    def logistic(st):
        model=joblib.load("MachineLearning/Logistic")
        input=[st]
        binarydata=feature_extraction.transform(input)
        result=model.predict(binarydata)
        if(result[0]==0):
            return spam
        elif(result[0]==1):
            return nospam
            

