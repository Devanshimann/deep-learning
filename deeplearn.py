import pandas as pd
import numpy as np 
from sklearn.linear_model import Perceptron
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
df=pd.read_excel("marks.xlsx")
from sklearn.model_selection import train_test_split
x=df.iloc[:,:-1]
y=df["placed"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
pc=Perceptron()
pc.fit(x_train,y_train)
pred=pc.predict(x_test)
print(pc.score(x_train,y_train)*100)
print(pc.score(x_test,y_test)*100)
from sklearn.metrics import accuracy_score
acuracy=accuracy_score(y_test,pred)
print(acuracy)
plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=pc)
plt.show()