#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


# In[2]:


fashion_mnist=tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[3]:


train_images.shape


# In[4]:


test_images.shape


# In[5]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[6]:


train_images=train_images/255.0
test_images=test_images/255.0


# In[7]:


plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[10]:


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')


# In[11]:


model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(128,activation= tf.keras.layers.LeakyReLU(alpha=0.3)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax'),
])


# In[12]:


model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[13]:


model.summary()


# In[14]:


model.fit(train_images, train_labels, epochs=3)


# In[15]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('\nTest accuracy:', test_acc)


# # what is the next step 
# 
# if an user enter to frontend -->
# shirt --> from the databse what ever the image is trained those imagew ill pop 
# data base recommane pullover --> jeans 
# 
# recommonda engine -->
# item based 
# produce based 
# 
# lets assure if front end user enter jeans -- 
# 
# 60 jeans -- customer may like or may like
# 
# 2month - 1 cusoterm busy the jean ( 60 jean are out dated ( fashion is old) 
# company grows or compnay loss 
# 
# compnay is lookin gor new fashion trend ( global market) 
# 
# AI CAN IMPLETE TO PRODUCT
# 
# project -->
# chatgpt ( new fashion trend in jeasn with manufact)
# messo ( dire manuse top in dife) 
# 
# daata 60 jean ( 600 ea)
# customer visiot webste new updat fashion 
# 
# 1day -- it grow (  customer use kie ) rate it 
# compnay grows
# AI help you or not help you 
# 
# you train the image in india dress 
# indina) mode
# 
# indian model will run in america, 
# model will urn is usa 
# 
# www. fahio.com
# 
# fasiho.=. c - in 
# fashion .com -- usa 
# amazon.com
# clouse 
# 
# amazon - sage maker
# snowflake (cloud) 
# 
# 
# most of the compnay they dont try to set up in theire own compay
# they alookin partner (new ai starup ventur) who can solve the business
