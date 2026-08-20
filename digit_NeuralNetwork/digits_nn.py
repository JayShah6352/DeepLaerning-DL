import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import  MLPClassifier ##Multi_leayer perceptron
from sklearn.neighbors import KNeighborsClassifier 

##Data preprossing start

digits=datasets.load_digits()
# print(digits.DESCR)
# print(digits.images)
# print(len(digits.target))
# print(digits.images.shape)##(1797,8,8)
# print(digits.target.shape)##(1797,)
images_with_labels=list(zip(digits.images,digits.target))
# print(images_with_labels[:2])
# for index,(image,label) in enumerate(images_with_labels[:6]):
#     plt.subplot(2,3,index + 1)
#     plt.imshow(image,cmap=plt.cm.gray_r)
#     plt.title("Training : %i"%label)
 # plt.show()

# n_sampls=len(digits.images)
# data=digits.images.reshape(n_sampls,-1)
# print(digits.images.shape)##(1797,8,8) 
# print(data.shape)##(1797,64) new shape
# print(data)
# data=data/255.
# print(data)

X,y=datasets.load_digits(return_X_y=True)
# print(X.shape)##(1797,64)
# print(y.shape)##(1797,)
# print(X[:2])
X=X/255.
# print(X[:2])

##Data preprossing end

model=MLPClassifier(hidden_layer_sizes=(50,),max_iter=300,random_state=1,verbose=1,learning_rate_init=.1)
model.fit(X,y)
# print("MLPClassifir: ",model.score(X,y))

KNeighborsClassifierMosel=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
# print("KNN classifier: ",model.score(X,y))

