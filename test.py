import numpy as np
a = np.array([[1,1,2],[2,1,2],[2,1,1]])

for i in range(3):
    if (a[i] == a[i]).any():
        print(a[i])
        print("yay")
    
    if (a[:, i] == a[:, i]).any():
        print(a[:, i])
        print("yey")