import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import PIL
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos',origin=dataset_url,cache_dir='.',untar=True)
# # print(data_dir)
data_dir="./datasets/flower_photos"
data_dir=pathlib.Path(data_dir)
# print(data_dir)
# print(list(data_dir.glob('*/*.jpg')))
image_count=len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

tulips=(list(data_dir.glob('tulips/*')))
# print(tulips[:5])

# PIL.Image.open(str(tulips[0])).show()

roses=(list(data_dir.glob('roses/*')))
# PIL.Image.open(str(roses[2])).show()

flowers_images_dict={
    'roses' : list(data_dir.glob('roses/*')),
    'daisy' : list(data_dir.glob('daisy/*')),
    'dandelion':list(data_dir.glob('dandelion/*')),
    'sunflowers':list(data_dir.glob('sunflowers/*')),
    'tulips':list(data_dir.glob('tulips/*'))
}
# print(flowers_images_dict['sunflowers'])

flowers_labels_dict={
    'roses' : 0,
    'daisy' :1,
    'dandelion':2,
    'sunflowers':3,
    'tulips':4
}

# print(flowers_images_dict['roses'][0])
img=cv2.imread(str(flowers_images_dict['roses'][0]))
# print(img)
# print(img.shape)
img=cv2.resize(img,(180,180))
# print(img.shape)##(180,180,3)

# for flower_name , images in flowers_images_dict.items():
#     print("flower name : ",flower_name,
#           "number of : " ,flower_name,
#           "is : ",len(images))
X=[]
y=[]
for flower_name , images in flowers_images_dict.items():
    for image in images:
        img=cv2.imread(str(image))
        resized_img=cv2.resize(img,(180,180))

        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
X=np.array(X)
y=np.array(y)
# print(X.shape)##(500,180,180,3)
# print(y.shape)##(500,)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
# print("X_train shape :",X_train.shape)##(375,180,180,3)
# print("X_test shape :",X_test.shape)##(125,180,180,3)
# print("y_train shape:",y_train.shape)##(375,)
# print("y_test shape:",y_test.shape)##(125,)

# print(X_train[0])
# print(X_test[0])
X_train_scaled=X_train/255
# print(X_train_scaled[0])
X_test_scaled=X_test/255
# print(X_test_scaled[0])

## Augmentaion

img_height=180
img_widh=180

data_augmentaion=tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal",input_shape=(img_height,img_widh,3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


num_classes=5
model=tf.keras.Sequential([
    data_augmentaion,
    tf.keras.layers.Conv2D(16,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(num_classes,activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(X_train_scaled,y_train,epochs=10)
model.evaluate(X_test_scaled,y_test)

predictons=model.predict(X_test_scaled)
# print(predictons)
# print(np.argmax(predictons[5]))
# print("Predicting class from all testing images,,,\n")

labels_flowers_dict = {
    0: 'roses',
    1: 'daisy',
    2: 'dandelion',
    3: 'sunflowers',
    4: 'tulips'
}

for i in range(1,len(X_test_scaled)):
#    print(np.argmax(predictions[i]))
    predicted_class = np.argmax(predictons[i])
    # print(predicted_class,"-->",labels_flowers_dict[predicted_class])

plt.title("Normal Image")
plt.axis('off')
plt.imshow(X[100])
plt.show()

plt.title("Augmentaion Image")
plt.axis('off')
plt.imshow(data_augmentaion(X)[100].numpy().astype('uint8'))
plt.show()