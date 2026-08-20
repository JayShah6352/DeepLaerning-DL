import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("data/sonar_dataset.csv",header=None)
# print(df[:5])
# print(df.shape)             ## (208, 61)
# print(df.isna().sum())
# print(df.columns)
# print(df[60].value_counts())

X = df.drop(60, axis=1)
y = df[60]

# print(X.shape)         ## (208, 60)
# print(y.shape)         ## (208,)

# print(y[:5]

y = pd.get_dummies(y,drop_first=True)
# print(y[:5])
# print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# print("X_train:", X_train.shape)        ## (156, 60)
# print("X_test :", X_test.shape)         ## (156, 1)
# print("y_train:", y_train.shape)        ## (52, 60)
# print("y_test :", y_test.shape)         ## (52, 1)

model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=20,batch_size=8)

model.evaluate(X_test,y_test)

y_pred=model.predict(X_test).reshape(-1)
print(y_pred[:5])
# print(y_pred.shape)##(52,1)
# flattened_arr=y_pred.reshape(-1)
# print(flattened_arr)
y_pred=np.round(y_pred)
# print(y_pred[:10])
# y_pred_classes = [np.argmax(i) for i in y_pred]

print(classification_report(y_test, y_pred))
