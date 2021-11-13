# Wenxin's Blog Post 5



**For Blog Post 5, we are gonna work on image classification by using Tensorflow.** To be more specifically, 

*First, we can apply Tensorflow Datasets so that we can divide it into sub-datasets such as training, validation, and test data sets.*


*Second, applying various of data augmentation, or layers, can effectively boost up our accuracy.*


*Third, applying pre-trained models is available by using transfer learning.*



**In this blog post, we are gonna create a machine larning algorithm. By applying this algorithm, we can distinguish different pictures, say , dogs and cats in this blog.**

## §1. Load Packages and Obtain Data

---

**Import all these instruments we want.**


```python
import os
from tensorflow.keras import utils
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

**In this step, we are going to get the data. By runing the code that we are given, we can get a sample data that contains labeled images of cats and dogs**


```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```

    Downloading data from https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
    68608000/68606236 [==============================] - 1s 0us/step
    68616192/68606236 [==============================] - 1s 0us/step
    Found 2000 files belonging to 2 classes.
    Found 1000 files belonging to 2 classes.



```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

**Here we are gonna write a function to do some simple visualization**

**Specifically, the function will return some visualization where there are three pictures of cats in the first row and three random pictures of dogs in the second row.**


```python
def Visualize(train_dataset):
  class_names = ["cat","dog"]       
  # Create a new class name containing cat and dog
  plt.figure(figsize=(10,10))
  # Create an empty figure with specific figuresize
  Cats = []
  Dogs = []    
  # Create two empty list to store figure matrix for cats and dogs
  Label1 = []
  Label2 = []
  # Create two lists to store the corresponding dogs and cats labels
  for images, labels in train_dataset.take(1):
  # Looping over images and labels in one batch
    for i in range(32):
  # Looping over every images and labels in one batch
      if labels[i] == 0 and len(Cats) < 3:
  # If it has label 0(cat), and less than 3 cats matrix in list Cats
        Cats.append(images[i])
  # Plug it into Cats
        Label1.append(0)
  # Append 0 to list Label1
      if labels[i] == 1 and len(Dogs) < 3:
  # If it has label 1(dog), and less than 3 dogs matrix in list Cats
        Dogs.append(images[i])
  # Plug it into Dogs
        Label2.append(1)
  # Append 1 to list Label2
  Total = Cats + Dogs   # Combine these two lists
  Label = Label1 + Label2  # Combine these two label lists

  for i in range(6):
    ax = plt.subplot(3,3,i+1)    # Looping over each subplot
    plt.imshow(Total[i].numpy().astype("uint8"))
    # Plot the corresponding picture
    plt.title(class_names[Label[i]])
    # make the label
    plt.axis("off")
```

**Let's try our function**


```python
Visualize(train_dataset)
```


    
![output_12_0.png](/images/output_12_0.png)
    


**Next, we first need to get an iterator called labels through the code below**


```python
labels_iterator= train_dataset.unbatch().map(lambda image, label: label).as_numpy_iterator()
```

**Then, we are going to compute the number of cat images (with label 0) and the number of dog images (with label 1) in the trainging data set**


```python
Cat_Num = 0   
Dog_Num = 0
# Create the cat and dog number counter

for LABELS in labels_iterator:
  if LABELS == 0:
    Cat_Num = Cat_Num + 1   
  # If label is 0, update cat_num
  if LABELS == 1:
    Dog_Num = Dog_Num + 1
  # If label is 1, update dog_num
print("The number of images in the training data with label 0 is " + str(Cat_Num))
print("The number of images in the training data with label 1 is " + str(Dog_Num))
```

    The number of images in the training data with label 0 is 1000
    The number of images in the training data with label 1 is 1000


## §2. First Model

**For the first model, what we want is to plug some layers that we've discussed in class into the model tf.keras.Sequential**

- Requirment: at least two Conv2D layers, at least two MaxPooling2D layers, at least one Flatten layer, at least one Dense layer, and at least one Dropout layer.

Name it as model1.


```python
model1 = models.Sequential([
    # For the input part
    # Set the first layer Conv2D have 32 kernels, 
    # With shape 3*3, 'relu' non-linear transformation
    # And the shape of input                      
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    # MaxPooling2D((2, 2)) make a size of 2 by 2 to find the max
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # The layer Dropout(0.2) reduce 20% of the units in each layer for during training
    # So then we will not get overfitting
    layers.Dropout(0.2),
    # The Flatten() convert two dimension data into one row
    layers.Flatten(),
    layers.Dense(10)                    
])
```

Briefly describe a few of the things you tried: 

1. I've tried to contain one Conv2D layers into my model1, but the resulting accuracy is pretty low. 

2. Then I attempted to add one more Conv2D layer and MaxPooling 2D layer, then the accuracy boosts up. 

3. As I increase the parameter of the Dense number, the accuracy increases as well

**Train model1**


```python
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model1.fit(train_dataset,
                    epochs=20,
                    validation_data=(validation_dataset))
```

    Epoch 1/20
    63/63 [==============================] - 36s 83ms/step - loss: 24.6366 - accuracy: 0.5115 - val_loss: 0.9028 - val_accuracy: 0.5322
    Epoch 2/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.7227 - accuracy: 0.6255 - val_loss: 0.8144 - val_accuracy: 0.5780
    Epoch 3/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.5570 - accuracy: 0.7130 - val_loss: 1.2076 - val_accuracy: 0.5408
    Epoch 4/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.4451 - accuracy: 0.7835 - val_loss: 1.1543 - val_accuracy: 0.5928
    Epoch 5/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.3518 - accuracy: 0.8375 - val_loss: 1.1424 - val_accuracy: 0.5780
    Epoch 6/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.3443 - accuracy: 0.8440 - val_loss: 1.6295 - val_accuracy: 0.5545
    Epoch 7/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.2796 - accuracy: 0.8750 - val_loss: 1.5578 - val_accuracy: 0.5817
    Epoch 8/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.2283 - accuracy: 0.9035 - val_loss: 1.5998 - val_accuracy: 0.5693
    Epoch 9/20
    63/63 [==============================] - 5s 76ms/step - loss: 0.1579 - accuracy: 0.9390 - val_loss: 1.8856 - val_accuracy: 0.5767
    Epoch 10/20
    63/63 [==============================] - 5s 78ms/step - loss: 0.1667 - accuracy: 0.9385 - val_loss: 1.8904 - val_accuracy: 0.5619
    Epoch 11/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.2128 - accuracy: 0.9230 - val_loss: 1.8088 - val_accuracy: 0.5606
    Epoch 12/20
    63/63 [==============================] - 5s 79ms/step - loss: 0.1489 - accuracy: 0.9370 - val_loss: 2.6636 - val_accuracy: 0.5458
    Epoch 13/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.1735 - accuracy: 0.9360 - val_loss: 2.0124 - val_accuracy: 0.5495
    Epoch 14/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.1434 - accuracy: 0.9565 - val_loss: 1.8996 - val_accuracy: 0.5619
    Epoch 15/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.1245 - accuracy: 0.9610 - val_loss: 2.3471 - val_accuracy: 0.5631
    Epoch 16/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.0868 - accuracy: 0.9705 - val_loss: 2.5780 - val_accuracy: 0.5792
    Epoch 17/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.1328 - accuracy: 0.9540 - val_loss: 2.6291 - val_accuracy: 0.5619
    Epoch 18/20
    63/63 [==============================] - 5s 77ms/step - loss: 0.1294 - accuracy: 0.9570 - val_loss: 2.4259 - val_accuracy: 0.5681
    Epoch 19/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.0854 - accuracy: 0.9700 - val_loss: 3.3113 - val_accuracy: 0.5705
    Epoch 20/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.0927 - accuracy: 0.9710 - val_loss: 2.8647 - val_accuracy: 0.5668


**We are going to make the plot of the accuracy in terms of the training and validation dataset**


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7ff47027abd0>




    
![output_24_1.png](/images/output_24_1.png)
    



**1. The accuracy of my model stabilized between 53% and 56% during training.**

**2. The accuracy of the baseline should be approximately 50%, so my accuracy is somewhere 3% to 6% higher.**

**3. Yes, my training accuracy is much higher than the validation accuracy as you may notice from the graph, so the overfitting may exist.**

## §3. Model with Data Augmentation

**In the step, we are gonna add some layers involved with data augumentation; specifically, adding data augumentation layers will allow modified copies of the same images**

### First, create a tf.keras.layers.RandomFlip() layer. 


```python
RandomFlip = tf.keras.layers.RandomFlip()
```

**Apply RandomFilp( ) and make plots**


```python
for image, _ in train_dataset.take(1):
  # Get the image data
  plt.figure(figsize=(10, 10))
  # Create an empty figure
  first_image = image[0]
  # Get the first_image
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
  # Looping over each subimage 
    augmented_image = RandomFlip(tf.expand_dims(first_image, 0))
  # Process each image by using RandomFlip
    plt.imshow(augmented_image[0] / 255)
  # make the figure
    plt.axis('off')
```


    
![output_31_0.png](/images/output_31_0.png)
    


### Next, create a tf.keras.layers.RandomRotation() layer


```python
RandomRotation = tf.keras.layers.RandomRotation(0.9)
```

**Apply RandomRotation( ) and make plots**


```python
for image, _ in train_dataset.take(1):
# Get the image data
  plt.figure(figsize=(10, 10))
# Create an empty figure
  first_image = image[0]
# Get the first_image
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
# Looping over each subimage 
    augmented_image = RandomRotation(tf.expand_dims(first_image, 0))
# Process each image by using RandomRotation
    plt.imshow(augmented_image[0] / 255)
# make the figure
    plt.axis('off')
```


    
![output_35_0.png](/images/output_35_0.png)
    


**Now we can simply add a RandomFlip( ) layer and a RandomRotation( ) layer to a tf.keras.models.Sequential and name it model2**


```python
model2 = models.Sequential([
    # add a RandomFlip( ) layer and a RandomRotation( ) layer
    layers.RandomFlip(),
    layers.RandomRotation(0.9),

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(10)                    
])
```

**Train model2**


```python
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model2.fit(train_dataset,
                    epochs=20,
                    validation_data=(validation_dataset))
```

    Epoch 1/20
    63/63 [==============================] - 7s 85ms/step - loss: 32.3707 - accuracy: 0.5190 - val_loss: 0.7445 - val_accuracy: 0.5099
    Epoch 2/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.7395 - accuracy: 0.5725 - val_loss: 0.7483 - val_accuracy: 0.5297
    Epoch 3/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.6911 - accuracy: 0.5670 - val_loss: 0.6974 - val_accuracy: 0.5594
    Epoch 4/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.6939 - accuracy: 0.5685 - val_loss: 0.6876 - val_accuracy: 0.5730
    Epoch 5/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.7064 - accuracy: 0.5760 - val_loss: 0.6793 - val_accuracy: 0.5644
    Epoch 6/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.6922 - accuracy: 0.5855 - val_loss: 0.7453 - val_accuracy: 0.6139
    Epoch 7/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.7073 - accuracy: 0.5745 - val_loss: 0.7068 - val_accuracy: 0.6027
    Epoch 8/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.6846 - accuracy: 0.5895 - val_loss: 0.6880 - val_accuracy: 0.5631
    Epoch 9/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.6694 - accuracy: 0.6000 - val_loss: 0.6848 - val_accuracy: 0.5817
    Epoch 10/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.6751 - accuracy: 0.5910 - val_loss: 0.6777 - val_accuracy: 0.5829
    Epoch 11/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.6759 - accuracy: 0.6040 - val_loss: 0.6705 - val_accuracy: 0.6262
    Epoch 12/20
    63/63 [==============================] - 5s 83ms/step - loss: 0.6768 - accuracy: 0.6170 - val_loss: 0.7089 - val_accuracy: 0.5582
    Epoch 13/20
    63/63 [==============================] - 5s 82ms/step - loss: 0.6683 - accuracy: 0.6200 - val_loss: 0.6842 - val_accuracy: 0.5916
    Epoch 14/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.6747 - accuracy: 0.6075 - val_loss: 0.6627 - val_accuracy: 0.6089
    Epoch 15/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.6671 - accuracy: 0.6175 - val_loss: 0.6715 - val_accuracy: 0.5928
    Epoch 16/20
    63/63 [==============================] - 5s 80ms/step - loss: 0.6717 - accuracy: 0.6060 - val_loss: 0.6855 - val_accuracy: 0.5903
    Epoch 17/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.6711 - accuracy: 0.6175 - val_loss: 0.6908 - val_accuracy: 0.6089
    Epoch 18/20
    63/63 [==============================] - 5s 81ms/step - loss: 0.6688 - accuracy: 0.6125 - val_loss: 0.6703 - val_accuracy: 0.6349
    Epoch 19/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.6584 - accuracy: 0.6085 - val_loss: 0.6755 - val_accuracy: 0.6040
    Epoch 20/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.6639 - accuracy: 0.6110 - val_loss: 0.7225 - val_accuracy: 0.5817


**Visualize the training history**


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7ff3faf13290>




    
![output_41_1.png](/images/output_41_1.png)
    


**1. The accuracy of my model stabilized between 54% and 62% during training.**

**2. The accuracy of that in the previous section should be around 53% to 56%, so my accuracy is somewhere 5% higher in this section.**

**3. No, my training accuracy is approximately the same as validation accuracy as you may notice from the graph, so the overfitting may not exist.**

## §4. Data Preprocessing

**In this part, we are going to make some simple transformations to the input so that it's easier and faster for our model to train**

**So the code below will create a layer called preprocessor and then incorprate the preprocessor layer into model3**


```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])
```

**Insert the preprocessor layer as the first layer of model3**


```python
model3 = models.Sequential([
    #Insert the preprocessor layer as the first layer of model3
    preprocessor,   
    layers.RandomFlip(),
    layers.RandomRotation(0.9),

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(10)                    
])
```

**Train model3**


```python
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model3.fit(train_dataset,
                    epochs=20,
                    validation_data=(validation_dataset))
```

    Epoch 1/20
    63/63 [==============================] - 7s 87ms/step - loss: 0.8699 - accuracy: 0.5025 - val_loss: 0.6869 - val_accuracy: 0.5705
    Epoch 2/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.6739 - accuracy: 0.5740 - val_loss: 0.6497 - val_accuracy: 0.5767
    Epoch 3/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.6321 - accuracy: 0.6340 - val_loss: 0.6092 - val_accuracy: 0.6547
    Epoch 4/20
    63/63 [==============================] - 6s 83ms/step - loss: 0.6304 - accuracy: 0.6370 - val_loss: 0.5935 - val_accuracy: 0.6745
    Epoch 5/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.6005 - accuracy: 0.6765 - val_loss: 0.6258 - val_accuracy: 0.6547
    Epoch 6/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.5879 - accuracy: 0.6900 - val_loss: 0.5655 - val_accuracy: 0.7030
    Epoch 7/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.6088 - accuracy: 0.6770 - val_loss: 0.6035 - val_accuracy: 0.6782
    Epoch 8/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5820 - accuracy: 0.6910 - val_loss: 0.5935 - val_accuracy: 0.6782
    Epoch 9/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.5618 - accuracy: 0.7270 - val_loss: 0.5881 - val_accuracy: 0.6955
    Epoch 10/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5590 - accuracy: 0.7145 - val_loss: 0.5761 - val_accuracy: 0.7079
    Epoch 11/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5602 - accuracy: 0.7060 - val_loss: 0.5539 - val_accuracy: 0.7203
    Epoch 12/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.5457 - accuracy: 0.7145 - val_loss: 0.6233 - val_accuracy: 0.6733
    Epoch 13/20
    63/63 [==============================] - 6s 85ms/step - loss: 0.5401 - accuracy: 0.7295 - val_loss: 0.5518 - val_accuracy: 0.7389
    Epoch 14/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.5427 - accuracy: 0.7170 - val_loss: 0.5893 - val_accuracy: 0.6832
    Epoch 15/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.5288 - accuracy: 0.7315 - val_loss: 0.5610 - val_accuracy: 0.7178
    Epoch 16/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.5164 - accuracy: 0.7445 - val_loss: 0.5781 - val_accuracy: 0.7116
    Epoch 17/20
    63/63 [==============================] - 6s 84ms/step - loss: 0.5330 - accuracy: 0.7305 - val_loss: 0.6198 - val_accuracy: 0.7005
    Epoch 18/20
    63/63 [==============================] - 6s 89ms/step - loss: 0.5226 - accuracy: 0.7355 - val_loss: 0.5507 - val_accuracy: 0.7215
    Epoch 19/20
    63/63 [==============================] - 6s 87ms/step - loss: 0.5197 - accuracy: 0.7400 - val_loss: 0.5583 - val_accuracy: 0.7215
    Epoch 20/20
    63/63 [==============================] - 6s 86ms/step - loss: 0.5133 - accuracy: 0.7535 - val_loss: 0.5689 - val_accuracy: 0.7339


**Visualize the training history**


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7ff478c2f650>




    
![output_51_1.png](/images/output_51_1.png)
    


**1. The accuracy of my model stabilized around 70% during training.**

**2. The accuracy of model1 is between 53% to 56%, so my accuracy is somewhere 17% higher in this section.**

**3. No, my training accuracy is approximately the same as validation accuracy as you may notice from the graph, so the overfitting may not exist.**

## §5. Transfer Learning

**In this part, we are going to incorporate some machine learning models that have already existed for our data. So basically, we first get a existing "base model" from the code below and then plug it into our new model**


```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
    9412608/9406464 [==============================] - 0s 0us/step
    9420800/9406464 [==============================] - 0s 0us/step


**Create model4 by adding the preprocessor layer we've derived before, the RandomFlip() and RandomRotation() from part 3, the base_model_layer above, and a Dense(2) layer at the end**


```python
model4 = models.Sequential([
    preprocessor, # adding the preprocessor layer we've derived before

    layers.RandomFlip(),
    layers.RandomRotation(0.9),
    # the RandomFlip() and RandomRotation() from part 3, 

    base_model,   # the base_model_layer above, 

    layers.GlobalMaxPooling2D(),
    layers.Dropout(0.2),

    layers.Dense(2)  # and a Dense(2) layer at the end 
              
])
```

**Here we show the summary and comment. As we can see, there are 2562 trainable parameters and 2257984 non-trainable parameters in this model.**


```python
model4.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     model (Functional)          (None, 160, 160, 3)       0         
                                                                     
     random_flip_3 (RandomFlip)  (None, 160, 160, 3)       0         
                                                                     
     random_rotation_3 (RandomRo  (None, 160, 160, 3)      0         
     tation)                                                         
                                                                     
     mobilenetv2_1.00_160 (Funct  (None, 5, 5, 1280)       2257984   
     ional)                                                          
                                                                     
     global_max_pooling2d (Globa  (None, 1280)             0         
     lMaxPooling2D)                                                  
                                                                     
     dropout_3 (Dropout)         (None, 1280)              0         
                                                                     
     dense_5 (Dense)             (None, 2)                 2562      
                                                                     
    =================================================================
    Total params: 2,260,546
    Trainable params: 2,562
    Non-trainable params: 2,257,984
    _________________________________________________________________


**Train model4 for 20 epochs**


```python
model4.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model4.fit(train_dataset,
                    epochs=20,
                    validation_data=(validation_dataset))
```

    Epoch 1/20
    63/63 [==============================] - 12s 117ms/step - loss: 1.3454 - accuracy: 0.7190 - val_loss: 0.2076 - val_accuracy: 0.9319
    Epoch 2/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.6024 - accuracy: 0.8495 - val_loss: 0.1630 - val_accuracy: 0.9530
    Epoch 3/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.4792 - accuracy: 0.8665 - val_loss: 0.0942 - val_accuracy: 0.9691
    Epoch 4/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.4028 - accuracy: 0.8950 - val_loss: 0.0910 - val_accuracy: 0.9666
    Epoch 5/20
    63/63 [==============================] - 6s 95ms/step - loss: 0.3839 - accuracy: 0.8910 - val_loss: 0.0829 - val_accuracy: 0.9678
    Epoch 6/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3579 - accuracy: 0.9000 - val_loss: 0.1085 - val_accuracy: 0.9604
    Epoch 7/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3347 - accuracy: 0.9005 - val_loss: 0.0919 - val_accuracy: 0.9678
    Epoch 8/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.3568 - accuracy: 0.8970 - val_loss: 0.0864 - val_accuracy: 0.9715
    Epoch 9/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3512 - accuracy: 0.9060 - val_loss: 0.0838 - val_accuracy: 0.9728
    Epoch 10/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.4262 - accuracy: 0.8880 - val_loss: 0.0918 - val_accuracy: 0.9703
    Epoch 11/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3137 - accuracy: 0.9045 - val_loss: 0.1353 - val_accuracy: 0.9604
    Epoch 12/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.3382 - accuracy: 0.8985 - val_loss: 0.1023 - val_accuracy: 0.9666
    Epoch 13/20
    63/63 [==============================] - 6s 95ms/step - loss: 0.2962 - accuracy: 0.9105 - val_loss: 0.0878 - val_accuracy: 0.9691
    Epoch 14/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.2932 - accuracy: 0.9080 - val_loss: 0.0909 - val_accuracy: 0.9678
    Epoch 15/20
    63/63 [==============================] - 6s 96ms/step - loss: 0.4026 - accuracy: 0.8960 - val_loss: 0.1061 - val_accuracy: 0.9703
    Epoch 16/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.2990 - accuracy: 0.9165 - val_loss: 0.1289 - val_accuracy: 0.9629
    Epoch 17/20
    63/63 [==============================] - 6s 96ms/step - loss: 0.3068 - accuracy: 0.9130 - val_loss: 0.1709 - val_accuracy: 0.9554
    Epoch 18/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.3699 - accuracy: 0.8940 - val_loss: 0.0967 - val_accuracy: 0.9703
    Epoch 19/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.3041 - accuracy: 0.9100 - val_loss: 0.0887 - val_accuracy: 0.9703
    Epoch 20/20
    63/63 [==============================] - 6s 96ms/step - loss: 0.3056 - accuracy: 0.9045 - val_loss: 0.0665 - val_accuracy: 0.9765


**Visualize the training history**


```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7ff3f7ffe910>




    
![output_63_1.png](/images/output_63_1.png)
    


**1. The accuracy of my model stabilized between 95% to 97% during training.**

**2. The accuracy of model1 is between 53% to 56%, so the accuracy is somewhere 42% higher in this section.**

**3. No, my training accuracy is actually lower than my validation accuracy as you may notice from the graph, so the overfitting may not exist.**

## §6. Score on Test Data

**Add one more dense layer into model4 so that we get the best validation accuracy. Name it as model5**


```python
model5 = models.Sequential([
    preprocessor,

    layers.RandomFlip(),
    layers.RandomRotation(0.9),

    base_model,

    layers.GlobalMaxPooling2D(),
    layers.Dropout(0.2),

    layers.Dense(128, activation='relu'),
    # Add one more dense layer
    layers.Dense(2)                    
])
```

**Train model5**


```python
model5.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

**Visualize the training history**


```python
history = model5.fit(train_dataset,
                    epochs=20,
                    validation_data=(validation_dataset))
```

    Epoch 1/20
    63/63 [==============================] - 11s 107ms/step - loss: 0.7216 - accuracy: 0.7995 - val_loss: 0.0984 - val_accuracy: 0.9653
    Epoch 2/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.2844 - accuracy: 0.8690 - val_loss: 0.1241 - val_accuracy: 0.9480
    Epoch 3/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.2270 - accuracy: 0.9005 - val_loss: 0.0673 - val_accuracy: 0.9715
    Epoch 4/20
    63/63 [==============================] - 6s 91ms/step - loss: 0.2214 - accuracy: 0.9040 - val_loss: 0.0952 - val_accuracy: 0.9579
    Epoch 5/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.2232 - accuracy: 0.8985 - val_loss: 0.0749 - val_accuracy: 0.9703
    Epoch 6/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1974 - accuracy: 0.9235 - val_loss: 0.0711 - val_accuracy: 0.9715
    Epoch 7/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.1854 - accuracy: 0.9250 - val_loss: 0.0579 - val_accuracy: 0.9790
    Epoch 8/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1907 - accuracy: 0.9220 - val_loss: 0.0550 - val_accuracy: 0.9851
    Epoch 9/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1872 - accuracy: 0.9245 - val_loss: 0.0539 - val_accuracy: 0.9814
    Epoch 10/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.2049 - accuracy: 0.9130 - val_loss: 0.0606 - val_accuracy: 0.9790
    Epoch 11/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1863 - accuracy: 0.9275 - val_loss: 0.0524 - val_accuracy: 0.9814
    Epoch 12/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1688 - accuracy: 0.9285 - val_loss: 0.0576 - val_accuracy: 0.9827
    Epoch 13/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1625 - accuracy: 0.9360 - val_loss: 0.0753 - val_accuracy: 0.9703
    Epoch 14/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1632 - accuracy: 0.9310 - val_loss: 0.0405 - val_accuracy: 0.9839
    Epoch 15/20
    63/63 [==============================] - 6s 94ms/step - loss: 0.1546 - accuracy: 0.9335 - val_loss: 0.0530 - val_accuracy: 0.9790
    Epoch 16/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1941 - accuracy: 0.9255 - val_loss: 0.0739 - val_accuracy: 0.9715
    Epoch 17/20
    63/63 [==============================] - 6s 92ms/step - loss: 0.1555 - accuracy: 0.9355 - val_loss: 0.0553 - val_accuracy: 0.9765
    Epoch 18/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1396 - accuracy: 0.9480 - val_loss: 0.0576 - val_accuracy: 0.9765
    Epoch 19/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1700 - accuracy: 0.9345 - val_loss: 0.0479 - val_accuracy: 0.9802
    Epoch 20/20
    63/63 [==============================] - 6s 93ms/step - loss: 0.1424 - accuracy: 0.9385 - val_loss: 0.0551 - val_accuracy: 0.9765



```python
model5.evaluate(test_dataset)
```

    6/6 [==============================] - 1s 71ms/step - loss: 0.0559 - accuracy: 0.9844





    [0.05593166872859001, 0.984375]



**Finally, we tried to evaluate the accuracy of model5 we just created, we get 0.9643 accuracy, which is awesome!!!**
