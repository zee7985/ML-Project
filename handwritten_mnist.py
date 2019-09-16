import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification

def distance(x1,x2):
    return np.sqrt((sum(x1-x2)**2))


def knn(x,y,querypt,k=5):
    
    val=[]
    m=x.shape[0]
    
    for i in range(m):
        d=distance(querypt,x[i])
        val.append((d,y[i]))
        
    val=sorted(val)
    
    #only nearest k 
    val=val[ :k]
    val=np.array(val)
    #print(val)
    
    new_val=np.unique((val[ : ,1]),return_counts=True)
    print(new_val)
    max_freq_index=new_val[1].argmax()
    pred=new_val[0][max_freq_index]
    return pred

#recognising handwritten digits using knn

df=pd.read_csv(r'C:\Users\zeesh\OneDrive\Desktop\dataset\mnist_test.csv')
print(df.shape)


data=df.values

x=data[  :,1: ]
y=data[  : ,0]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)


#visualise some image example
print(xtrain.shape,ytrain.shape)
print(xtest.shape)


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img)
    plt.show()



pred=knn(xtrain,ytrain,xtest[2],k=5)
print(pred)



drawImg(xtest[2])
print(ytest[2])