import numpy as np
import pandas as pd
import os


def perceptron(X, labels, eta=.5, obs=None, w=None, mistakes=None):
    w = np.ones(X.shape[1]).reshape(-1,1)
    mistakes = []
    indexes = range(X.shape[0])
    
    m = 1
    iteration = 1
    while m!= 0:
        m = 0
        
        print(f'Pass number: {iteration}')
        for i in indexes:
            x = X[i].reshape(-1,1)
            y = labels[i]

            w_x = np.dot(w.T,x)
            if w_x >= 0:
                yhat = 1
            else:
                yhat = 0
            if yhat != y:
                print(f'     Weight update: {w.reshape(w.shape[0],).tolist()} -> ', end='')
                update = (y - yhat)
                w += eta*update*x
                print(f'{w.reshape(w.shape[0],).tolist()}')
                m += 1
        mistakes.append(m)
        
        # if no mistakes during pass - break
        print(f'     Total number of mistakes in pass: {m}\n')
        if m == 0:
            break
        iteration += 1
      
    w = w.reshape(w.shape[0],1)    
    return w, mistakes

# Load data
os.chdir(r'C:\Users\josep\python_files\CptS_540\assignment_10')
df = pd.read_csv(r'havefun_df.csv')

# X
X = df.iloc[:,:-1]
intercept = pd.Series([1 for _ in range(len(X))])
X.insert(0, 'intercept', intercept)
X = np.array(X)

# Labels
labels = np.array(df[['HaveFun']])

# Train model
w, mistakes = perceptron(X, labels)
print(f'w = {w.reshape(w.shape[0],).tolist()}')
