## Basic nural network
import numpy as np
### Input
X=np.array([1,2])
## Intal Weight
W=np.array([0.5,0.3])
##Bias
b=0.1
##Target Output
y=1
##Activation Function
def sigmoid(X):
    return 1/(1+np.exp(-X))
##Neural network ==> Forward propogation
z=np.dot(X,W)+b
output=sigmoid(z)
print("output=",output)
error=y-output
print("Error=",error)
if output >= 0.5:
    output = 1
else:
    output=0
print("output=",output)
##INput==>weight*input(z)==>Activation Function ==> output ==> error
## neural network==> Backward propogation
##error==>output==>Activation ==> z ==> weight and bias

sigmoid_derivtive=output*(1-output)
delta=error*sigmoid_derivtive
learning_rate=0.1
W=W+learning_rate*delta*X
b=b+learning_rate*delta
print("\n Updated Wight=",W)
print("\n Updated Bias=",b)
# m=m+learning_rate*dm
##Forward propogation
z=np.dot(X,W)+b
new_output=sigmoid(z)
print("\n output after training=",new_output)
