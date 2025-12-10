import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
df=pd.read_csv("churn_dataset_1000.csv")
sc= StandardScaler()
df.isnull().sum()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
xx=pd.DataFrame(sc.fit_transform(x_train),columns=x_train.columns)
x2=pd.DataFrame(sc.fit_transform(x_test),columns=x_test.columns)
ann=Sequential()
print(xx.shape)
ann.add(Dense(6,activation="relu",input_dim=8))
ann.add(Dense(4,activation="relu"))
ann.add(Dense(2,activation="relu"))
ann.add(Dense(1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy")
ann.fit(xx,y_train,batch_size=10,epochs=10)
pred=ann.predict(x2)
new_pred=[]
for i in pred:
    if i[0]>0.5:
        new_pred.append(1)
    else:
        new_pred.append(0)
from sklearn.metrics import accuracy_score
acc= accuracy_score(new_pred,y_test)
print(acc*100)
pred1=ann.predict(xx)
new_pred1=[]
for i in pred1:
    if i[0]>0.5:
        new_pred1.append(1)
    else:
        new_pred1.append(0)

acc1=accuracy_score(y_train,new_pred1)
print(acc1*100)