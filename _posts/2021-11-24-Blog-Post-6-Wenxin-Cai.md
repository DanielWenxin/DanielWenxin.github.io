# Wenxin's Blog-Post-6



```python
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras

# requires update to tensorflow 2.4
# >>> conda activate PIC16B
# >>> pip install tensorflow==2.4
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for embedding viz
import plotly.express as px 
import plotly.io as pio
pio.templates.default = "plotly_white"
```

# §1. Acquire Training Data

Each row of the data corresponds to an article. The title column gives the title of the article, while the text column gives the full article text. The final column, called fake, is 0 if the article is true and 1 if the article contains fake news


```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
df  = pd.read_csv(train_url)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# §2. Make a Dataset

We wanna write a function called make_dataset, where the function should do two things

1. Remove stopwords from the article text and title

2. Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form (title, text), and the output should consist only of the fake column. 


```python
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop = stopwords.words('english')
# inport the set of words we tend to consider as stopwords
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.



```python
def make_dataset(df):
  df["title"] = df["title"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  # remove the stopwords in title column
  df["text"] = df["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  # remove the stopwords in text column

  data = tf.data.Dataset.from_tensor_slices(
    (
        {
            "title" : df[["title"]], 
            "text" : df[["text"]],
        }, 
        {
            "fake" : df[["fake"]]
        }
    )
)
  # Construct and return a tf.data.Dataset with two inputs and one output. 
  # The input should be of the form (title, text), and the output should consist only of the fake column
  data = data.batch(100)
  # Batch your Dataset prior to returning it
  return data

```


```python
Dataset = make_dataset(df)  # Plug our original data into Dataset
```

#### Validation Data

After you’ve constructed your primary Dataset, split of 20% of it to use for validation.


```python
train_size = int(0.8*len(Dataset))
train = Dataset.take(train_size)   # Split 80% of our data into train dataset
val   = Dataset.skip(train_size)   # Set the 20% of our data into train dataset
len(train), len(val)
```




    (180, 45)



#### Base Rate

The base rate refers to the accuracy of a model that always makes the same guess (for example, such a model might always say “fake news!”). Determine the base rate for this data set by examining the labels on the training set.


```python
labels_iterator= train.unbatch().map(lambda image,fake: fake).as_numpy_iterator()
True_Article = 0
Fake_Article = 0
for LABEL in labels_iterator:
    if LABEL["fake"]==0:
        True_Article=True_Article+1
    else:
        Fake_Article=Fake_Article+1
print(True_Article)
print(Fake_Article)
```

    8603
    9397



```python
base_rate = 9397 / (8603+9397)
print(base_rate)
```

    0.5220555555555556


As we can see, the base rate is approximately 0.522

## §3. Create Models

#### First Model

In the first model, we use only the article title as an input.


```python
size_vocabulary = 2000

# Here is the function for text standardization
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)       # convert into lowercast 
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    # remove punctuation and some other elements
    return no_punctuation 

# Here is the function for text vectorization
vectorize_layer = TextVectorization(
    standardize=standardization, # Plug in the standardization funtion
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
# Here is the vectorize layer
```


```python
title_input = keras.Input(
    shape = (1,),
    name = "title",
    dtype = "string"
)

# Set up the title input
```


```python
title_features = vectorize_layer(title_input)
title_features = layers.Embedding(size_vocabulary, 3, name = "embedding")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(16, activation='relu')(title_features)

# Set up the title layers as we discussed in the lecture
```


```python
output = layers.Dense(2, name = "fake")(title_features)
# Set up the output
```


```python
model1 = keras.Model(
    inputs = title_input,
    outputs = output
)
# In the first model, we should use only the article title as an input
```


```python
model1.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 500)              0         
     torization)                                                     
                                                                     
     embedding (Embedding)       (None, 500, 3)            6000      
                                                                     
     dropout (Dropout)           (None, 500, 3)            0         
                                                                     
     global_average_pooling1d (G  (None, 3)                0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 3)                 0         
                                                                     
     dense (Dense)               (None, 16)                64        
                                                                     
     fake (Dense)                (None, 2)                 34        
                                                                     
    =================================================================
    Total params: 6,098
    Trainable params: 6,098
    Non-trainable params: 0
    _________________________________________________________________



```python
keras.utils.plot_model(model1)  # Plot the model
```




    
![output_27_0.png](/images/output_27_0.png)    




```python
model1.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
# Train model1
history = model1.fit(train,
                    validation_data=val,
                    epochs = 30)
```

    Epoch 1/30


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning:
    
    Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
    


    180/180 [==============================] - 4s 11ms/step - loss: 0.6921 - accuracy: 0.5182 - val_loss: 0.6908 - val_accuracy: 0.5266
    Epoch 2/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.6891 - accuracy: 0.5318 - val_loss: 0.6824 - val_accuracy: 0.5278
    Epoch 3/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.6620 - accuracy: 0.6891 - val_loss: 0.6271 - val_accuracy: 0.7932
    Epoch 4/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.5651 - accuracy: 0.8765 - val_loss: 0.4911 - val_accuracy: 0.9400
    Epoch 5/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.4249 - accuracy: 0.9246 - val_loss: 0.3516 - val_accuracy: 0.9470
    Epoch 6/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.3165 - accuracy: 0.9327 - val_loss: 0.2595 - val_accuracy: 0.9526
    Epoch 7/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.2456 - accuracy: 0.9423 - val_loss: 0.2000 - val_accuracy: 0.9600
    Epoch 8/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.2007 - accuracy: 0.9505 - val_loss: 0.1622 - val_accuracy: 0.9643
    Epoch 9/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1706 - accuracy: 0.9539 - val_loss: 0.1376 - val_accuracy: 0.9710
    Epoch 10/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1503 - accuracy: 0.9566 - val_loss: 0.1196 - val_accuracy: 0.9724
    Epoch 11/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1325 - accuracy: 0.9626 - val_loss: 0.1051 - val_accuracy: 0.9737
    Epoch 12/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1209 - accuracy: 0.9630 - val_loss: 0.0944 - val_accuracy: 0.9748
    Epoch 13/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1093 - accuracy: 0.9669 - val_loss: 0.0906 - val_accuracy: 0.9777
    Epoch 14/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1029 - accuracy: 0.9694 - val_loss: 0.0807 - val_accuracy: 0.9775
    Epoch 15/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0971 - accuracy: 0.9696 - val_loss: 0.0757 - val_accuracy: 0.9786
    Epoch 16/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0897 - accuracy: 0.9722 - val_loss: 0.0722 - val_accuracy: 0.9800
    Epoch 17/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0851 - accuracy: 0.9717 - val_loss: 0.0691 - val_accuracy: 0.9802
    Epoch 18/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0828 - accuracy: 0.9723 - val_loss: 0.0663 - val_accuracy: 0.9802
    Epoch 19/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0773 - accuracy: 0.9751 - val_loss: 0.0622 - val_accuracy: 0.9804
    Epoch 20/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0762 - accuracy: 0.9738 - val_loss: 0.0612 - val_accuracy: 0.9809
    Epoch 21/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0730 - accuracy: 0.9753 - val_loss: 0.0578 - val_accuracy: 0.9807
    Epoch 22/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0721 - accuracy: 0.9748 - val_loss: 0.0564 - val_accuracy: 0.9807
    Epoch 23/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0699 - accuracy: 0.9748 - val_loss: 0.0543 - val_accuracy: 0.9811
    Epoch 24/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0681 - accuracy: 0.9764 - val_loss: 0.0533 - val_accuracy: 0.9816
    Epoch 25/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0642 - accuracy: 0.9773 - val_loss: 0.0540 - val_accuracy: 0.9813
    Epoch 26/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0645 - accuracy: 0.9771 - val_loss: 0.0517 - val_accuracy: 0.9811
    Epoch 27/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0626 - accuracy: 0.9768 - val_loss: 0.0532 - val_accuracy: 0.9813
    Epoch 28/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0603 - accuracy: 0.9782 - val_loss: 0.0497 - val_accuracy: 0.9822
    Epoch 29/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0594 - accuracy: 0.9787 - val_loss: 0.0518 - val_accuracy: 0.9818
    Epoch 30/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0578 - accuracy: 0.9795 - val_loss: 0.0484 - val_accuracy: 0.9822



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f37af4ab350>




    
![output_30_1.png](/images/output_30_1.png)
    


As we can see, the accuracy of model1 is approximately 98%. And there is no apparent overfitting observed.

## Second Model

In the second model, we use only the article text as an input.


```python
# Same as first model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

# Same as first model
vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```


```python
text_input = keras.Input(
    shape = (1,),
    name = "text",
    dtype = "string"
)

# Set up the title input
```


```python
text_features = vectorize_layer(text_input)
text_features = layers.Embedding(size_vocabulary, 3, name = "embedding")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(16, activation='relu')(text_features)
# Set up the text layers as we discussed in the lecture
```


```python
output = layers.Dense(2, name = "fake")(text_features)
# Set up the output
```


```python
model2 = keras.Model(
    inputs = text_input,
    outputs = output
)
# In the second model, we should use only the article text as an input
```


```python
model2.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text (InputLayer)           [(None, 1)]               0         
                                                                     
     text_vectorization_1 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding (Embedding)       (None, 500, 3)            6000      
                                                                     
     dropout_2 (Dropout)         (None, 500, 3)            0         
                                                                     
     global_average_pooling1d_1   (None, 3)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_3 (Dropout)         (None, 3)                 0         
                                                                     
     dense_1 (Dense)             (None, 16)                64        
                                                                     
     fake (Dense)                (None, 2)                 34        
                                                                     
    =================================================================
    Total params: 6,098
    Trainable params: 6,098
    Non-trainable params: 0
    _________________________________________________________________



```python
keras.utils.plot_model(model2)    # Plot model2
```




    
![output_40_0.png](/images/output_40_0.png)      
    




```python
model2.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
# Train model2
history = model2.fit(train,
                    validation_data=val,
                    epochs = 30)

```

    Epoch 1/30


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning:
    
    Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
    


    180/180 [==============================] - 4s 18ms/step - loss: 0.6741 - accuracy: 0.6317 - val_loss: 0.6276 - val_accuracy: 0.9155
    Epoch 2/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.5265 - accuracy: 0.8980 - val_loss: 0.4062 - val_accuracy: 0.9400
    Epoch 3/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.3366 - accuracy: 0.9301 - val_loss: 0.2607 - val_accuracy: 0.9512
    Epoch 4/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.2394 - accuracy: 0.9441 - val_loss: 0.1970 - val_accuracy: 0.9600
    Epoch 5/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1938 - accuracy: 0.9534 - val_loss: 0.1630 - val_accuracy: 0.9631
    Epoch 6/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1626 - accuracy: 0.9591 - val_loss: 0.1414 - val_accuracy: 0.9670
    Epoch 7/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1430 - accuracy: 0.9638 - val_loss: 0.1257 - val_accuracy: 0.9685
    Epoch 8/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1295 - accuracy: 0.9666 - val_loss: 0.1149 - val_accuracy: 0.9715
    Epoch 9/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1178 - accuracy: 0.9689 - val_loss: 0.1057 - val_accuracy: 0.9721
    Epoch 10/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1107 - accuracy: 0.9691 - val_loss: 0.1005 - val_accuracy: 0.9742
    Epoch 11/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1006 - accuracy: 0.9727 - val_loss: 0.0976 - val_accuracy: 0.9751
    Epoch 12/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0955 - accuracy: 0.9739 - val_loss: 0.0920 - val_accuracy: 0.9766
    Epoch 13/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0896 - accuracy: 0.9741 - val_loss: 0.0907 - val_accuracy: 0.9757
    Epoch 14/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0863 - accuracy: 0.9760 - val_loss: 0.0847 - val_accuracy: 0.9782
    Epoch 15/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0794 - accuracy: 0.9778 - val_loss: 0.0872 - val_accuracy: 0.9762
    Epoch 16/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0747 - accuracy: 0.9788 - val_loss: 0.0816 - val_accuracy: 0.9786
    Epoch 17/30
    180/180 [==============================] - 3s 18ms/step - loss: 0.0718 - accuracy: 0.9803 - val_loss: 0.0821 - val_accuracy: 0.9777
    Epoch 18/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0688 - accuracy: 0.9809 - val_loss: 0.0797 - val_accuracy: 0.9804
    Epoch 19/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0654 - accuracy: 0.9820 - val_loss: 0.0814 - val_accuracy: 0.9789
    Epoch 20/30
    180/180 [==============================] - 3s 18ms/step - loss: 0.0637 - accuracy: 0.9818 - val_loss: 0.0771 - val_accuracy: 0.9798
    Epoch 21/30
    180/180 [==============================] - 3s 18ms/step - loss: 0.0601 - accuracy: 0.9828 - val_loss: 0.0755 - val_accuracy: 0.9800
    Epoch 22/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0562 - accuracy: 0.9833 - val_loss: 0.0773 - val_accuracy: 0.9804
    Epoch 23/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0556 - accuracy: 0.9847 - val_loss: 0.0790 - val_accuracy: 0.9802
    Epoch 24/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0535 - accuracy: 0.9838 - val_loss: 0.0780 - val_accuracy: 0.9807
    Epoch 25/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0534 - accuracy: 0.9834 - val_loss: 0.0759 - val_accuracy: 0.9813
    Epoch 26/30
    180/180 [==============================] - 3s 18ms/step - loss: 0.0491 - accuracy: 0.9846 - val_loss: 0.0757 - val_accuracy: 0.9818
    Epoch 27/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0473 - accuracy: 0.9855 - val_loss: 0.0745 - val_accuracy: 0.9820
    Epoch 28/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0457 - accuracy: 0.9857 - val_loss: 0.0789 - val_accuracy: 0.9827
    Epoch 29/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0451 - accuracy: 0.9862 - val_loss: 0.0741 - val_accuracy: 0.9811
    Epoch 30/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0434 - accuracy: 0.9859 - val_loss: 0.0761 - val_accuracy: 0.9811



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f372cfefb90>




![output_43_1.png](/images/output_43_1.png)    
    


As we can see, the accuracy of model2 is approximately 98%.
And there is no apparent overfitting observed.

## Third Model

In the third model, we use both the article title and the article text as input.


```python
# same layer from model1
title_features = vectorize_layer(title_input)
title_features = layers.Embedding(size_vocabulary, 3, name = "embeddingTITLE")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)
```


```python
# same layer from model2
text_features = vectorize_layer(text_input)
text_features = layers.Embedding(size_vocabulary, 3, name = "embeddingTEXT")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)
```


```python
main = layers.concatenate([title_features, text_features], axis = 1)
# concatenate layers from model1 and model2
```


```python
main = layers.Dense(32, activation='relu')(main)
# Add one output layer
output = layers.Dense(2, name = "fake")(main)
# Set up the output layer
```


```python
model3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = output
)
# In the third model, we use both the article title and the article text as input
```


```python
model3.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     title (InputLayer)             [(None, 1)]          0           []                               
                                                                                                      
     text (InputLayer)              [(None, 1)]          0           []                               
                                                                                                      
     text_vectorization_1 (TextVect  (None, 500)         0           ['title[0][0]',                  
     orization)                                                       'text[0][0]']                   
                                                                                                      
     embeddingTITLE (Embedding)     (None, 500, 3)       6000        ['text_vectorization_1[1][0]']   
                                                                                                      
     embeddingTEXT (Embedding)      (None, 500, 3)       6000        ['text_vectorization_1[2][0]']   
                                                                                                      
     dropout_4 (Dropout)            (None, 500, 3)       0           ['embeddingTITLE[0][0]']         
                                                                                                      
     dropout_6 (Dropout)            (None, 500, 3)       0           ['embeddingTEXT[0][0]']          
                                                                                                      
     global_average_pooling1d_2 (Gl  (None, 3)           0           ['dropout_4[0][0]']              
     obalAveragePooling1D)                                                                            
                                                                                                      
     global_average_pooling1d_3 (Gl  (None, 3)           0           ['dropout_6[0][0]']              
     obalAveragePooling1D)                                                                            
                                                                                                      
     dropout_5 (Dropout)            (None, 3)            0           ['global_average_pooling1d_2[0][0
                                                                     ]']                              
                                                                                                      
     dropout_7 (Dropout)            (None, 3)            0           ['global_average_pooling1d_3[0][0
                                                                     ]']                              
                                                                                                      
     dense_2 (Dense)                (None, 32)           128         ['dropout_5[0][0]']              
                                                                                                      
     dense_3 (Dense)                (None, 32)           128         ['dropout_7[0][0]']              
                                                                                                      
     concatenate (Concatenate)      (None, 64)           0           ['dense_2[0][0]',                
                                                                      'dense_3[0][0]']                
                                                                                                      
     dense_4 (Dense)                (None, 32)           2080        ['concatenate[0][0]']            
                                                                                                      
     fake (Dense)                   (None, 2)            66          ['dense_4[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 14,402
    Trainable params: 14,402
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
keras.utils.plot_model(model3)
# Plot the model
```




    
![output_53_0.png](/images/output_53_0.png)    
    




```python
model3.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
# Train model3
history = model3.fit(train,
                    validation_data=val,
                    epochs = 30)

```

    Epoch 1/30
    180/180 [==============================] - 5s 22ms/step - loss: 0.6376 - accuracy: 0.6484 - val_loss: 0.4031 - val_accuracy: 0.9144
    Epoch 2/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.2422 - accuracy: 0.9338 - val_loss: 0.1554 - val_accuracy: 0.9620
    Epoch 3/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.1465 - accuracy: 0.9570 - val_loss: 0.1151 - val_accuracy: 0.9685
    Epoch 4/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.1174 - accuracy: 0.9657 - val_loss: 0.0971 - val_accuracy: 0.9717
    Epoch 5/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0964 - accuracy: 0.9719 - val_loss: 0.0859 - val_accuracy: 0.9759
    Epoch 6/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0852 - accuracy: 0.9751 - val_loss: 0.0791 - val_accuracy: 0.9777
    Epoch 7/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0763 - accuracy: 0.9772 - val_loss: 0.0718 - val_accuracy: 0.9793
    Epoch 8/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0611 - accuracy: 0.9828 - val_loss: 0.0673 - val_accuracy: 0.9804
    Epoch 9/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0507 - accuracy: 0.9870 - val_loss: 0.0548 - val_accuracy: 0.9854
    Epoch 10/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0385 - accuracy: 0.9903 - val_loss: 0.0465 - val_accuracy: 0.9879
    Epoch 11/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0295 - accuracy: 0.9926 - val_loss: 0.0395 - val_accuracy: 0.9881
    Epoch 12/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0262 - accuracy: 0.9926 - val_loss: 0.0332 - val_accuracy: 0.9899
    Epoch 13/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0205 - accuracy: 0.9940 - val_loss: 0.0286 - val_accuracy: 0.9915
    Epoch 14/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0181 - accuracy: 0.9946 - val_loss: 0.0292 - val_accuracy: 0.9915
    Epoch 15/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0158 - accuracy: 0.9954 - val_loss: 0.0263 - val_accuracy: 0.9924
    Epoch 16/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0137 - accuracy: 0.9959 - val_loss: 0.0261 - val_accuracy: 0.9928
    Epoch 17/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0123 - accuracy: 0.9959 - val_loss: 0.0247 - val_accuracy: 0.9933
    Epoch 18/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0129 - accuracy: 0.9963 - val_loss: 0.0247 - val_accuracy: 0.9926
    Epoch 19/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0130 - accuracy: 0.9958 - val_loss: 0.0230 - val_accuracy: 0.9933
    Epoch 20/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0117 - accuracy: 0.9964 - val_loss: 0.0225 - val_accuracy: 0.9933
    Epoch 21/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0110 - accuracy: 0.9962 - val_loss: 0.0237 - val_accuracy: 0.9930
    Epoch 22/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0086 - accuracy: 0.9973 - val_loss: 0.0234 - val_accuracy: 0.9939
    Epoch 23/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0097 - accuracy: 0.9964 - val_loss: 0.0276 - val_accuracy: 0.9912
    Epoch 24/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0084 - accuracy: 0.9969 - val_loss: 0.0253 - val_accuracy: 0.9935
    Epoch 25/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0073 - accuracy: 0.9978 - val_loss: 0.0262 - val_accuracy: 0.9926
    Epoch 26/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0085 - accuracy: 0.9974 - val_loss: 0.0281 - val_accuracy: 0.9924
    Epoch 27/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0099 - accuracy: 0.9964 - val_loss: 0.0209 - val_accuracy: 0.9937
    Epoch 28/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0081 - accuracy: 0.9971 - val_loss: 0.0250 - val_accuracy: 0.9937
    Epoch 29/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0075 - accuracy: 0.9974 - val_loss: 0.0264 - val_accuracy: 0.9935
    Epoch 30/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0078 - accuracy: 0.9978 - val_loss: 0.0223 - val_accuracy: 0.9939



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f372f822d50>




    
![output_56_1.png](/images/output_56_1.png)      
    


As we can see, the accuracy of model3 is close to 100%.
And there is no apparent overfitting observed. That's awesome!

# §4. Model Evaluation

Now we’ll test your model performance on unseen test data. For this part, we'll use model3.


```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
```


```python
test_df  = pd.read_csv(test_url)    # Derive the test dataframe
```


```python
test_df.head()     
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>420</td>
      <td>CNN And MSNBC Destroy Trump, Black Out His Fa...</td>
      <td>Donald Trump practically does something to cri...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14902</td>
      <td>Exclusive: Kremlin tells companies to deliver ...</td>
      <td>The Kremlin wants good news.  The Russian lead...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>322</td>
      <td>Golden State Warriors Coach Just WRECKED Trum...</td>
      <td>On Saturday, the man we re forced to call  Pre...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16108</td>
      <td>Putin opens monument to Stalin's victims, diss...</td>
      <td>President Vladimir Putin inaugurated a monumen...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10304</td>
      <td>BREAKING: DNC HACKER FIRED For Bank Fraud…Blam...</td>
      <td>Apparently breaking the law and scamming the g...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df2 = make_dataset(test_df)     # convert this data using the make_dataset function you defined in Part §2
```


```python
model3.evaluate(test_df2)    # evaluate model3 on the data
```

    225/225 [==============================] - 3s 12ms/step - loss: 0.0363 - accuracy: 0.9916





    [0.03625625744462013, 0.9915809035301208]



As we can see, the accuracy is close to 100%. That's awesome!

# §5. Embedding Visualization

Visualize and comment on the embedding that our model learned


```python
weights = 0.5*(model3.get_layer('embeddingTITLE').get_weights()[0]) + 0.5*(model3.get_layer('embeddingTEXT').get_weights()[0])
# get the weights from the embedding layer
vocab = vectorize_layer.get_vocabulary()  # get the vocabulary
```


```python
from sklearn.decomposition import PCA   # import PCA
pca = PCA(n_components=2)               # Convert our data into 2 dimension
weights = pca.fit_transform(weights)
```


```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
# Now we'll make a data frame from our results:
```


```python
import plotly.express as px 
from plotly.io import write_html
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")


write_html(fig, "Embed.html")
# Ready to plot!
```

This embedding seems to have learned some reasonable associations.

For example, we see that words like "London", "Spain", 'China','Japan','turkey', and "capital" are relatively close to each other. So are "to", "for", and "of", as well as "century", "21st", and "daily."
