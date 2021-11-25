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
    if LABEL==0:
        True_Article=True_Article+1
    else:
        Fake_Article=Fake_Article+1
print(True_Article)
print(Fake_Article)
```

    0
    18000



```python
Fake_Article
```




    18000



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
    


    180/180 [==============================] - 4s 11ms/step - loss: 0.6914 - accuracy: 0.5172 - val_loss: 0.6889 - val_accuracy: 0.5266
    Epoch 2/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.6811 - accuracy: 0.5807 - val_loss: 0.6665 - val_accuracy: 0.5846
    Epoch 3/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.6337 - accuracy: 0.7715 - val_loss: 0.5846 - val_accuracy: 0.9045
    Epoch 4/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.5184 - accuracy: 0.8897 - val_loss: 0.4418 - val_accuracy: 0.9445
    Epoch 5/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.3861 - accuracy: 0.9272 - val_loss: 0.3191 - val_accuracy: 0.9494
    Epoch 6/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.2913 - accuracy: 0.9397 - val_loss: 0.2399 - val_accuracy: 0.9544
    Epoch 7/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.2319 - accuracy: 0.9456 - val_loss: 0.1904 - val_accuracy: 0.9622
    Epoch 8/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1935 - accuracy: 0.9507 - val_loss: 0.1555 - val_accuracy: 0.9674
    Epoch 9/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1627 - accuracy: 0.9570 - val_loss: 0.1307 - val_accuracy: 0.9715
    Epoch 10/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1431 - accuracy: 0.9616 - val_loss: 0.1129 - val_accuracy: 0.9737
    Epoch 11/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1265 - accuracy: 0.9643 - val_loss: 0.1003 - val_accuracy: 0.9746
    Epoch 12/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1179 - accuracy: 0.9652 - val_loss: 0.0912 - val_accuracy: 0.9757
    Epoch 13/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.1066 - accuracy: 0.9685 - val_loss: 0.0875 - val_accuracy: 0.9773
    Epoch 14/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0989 - accuracy: 0.9696 - val_loss: 0.0808 - val_accuracy: 0.9775
    Epoch 15/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0926 - accuracy: 0.9712 - val_loss: 0.0725 - val_accuracy: 0.9795
    Epoch 16/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0888 - accuracy: 0.9712 - val_loss: 0.0683 - val_accuracy: 0.9807
    Epoch 17/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0825 - accuracy: 0.9737 - val_loss: 0.0665 - val_accuracy: 0.9807
    Epoch 18/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0809 - accuracy: 0.9732 - val_loss: 0.0643 - val_accuracy: 0.9807
    Epoch 19/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0784 - accuracy: 0.9744 - val_loss: 0.0600 - val_accuracy: 0.9809
    Epoch 20/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0745 - accuracy: 0.9747 - val_loss: 0.0577 - val_accuracy: 0.9807
    Epoch 21/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0685 - accuracy: 0.9766 - val_loss: 0.0577 - val_accuracy: 0.9813
    Epoch 22/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0693 - accuracy: 0.9757 - val_loss: 0.0552 - val_accuracy: 0.9813
    Epoch 23/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0673 - accuracy: 0.9765 - val_loss: 0.0528 - val_accuracy: 0.9829
    Epoch 24/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0646 - accuracy: 0.9768 - val_loss: 0.0576 - val_accuracy: 0.9809
    Epoch 25/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0634 - accuracy: 0.9776 - val_loss: 0.0527 - val_accuracy: 0.9816
    Epoch 26/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0608 - accuracy: 0.9786 - val_loss: 0.0513 - val_accuracy: 0.9818
    Epoch 27/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0591 - accuracy: 0.9801 - val_loss: 0.0516 - val_accuracy: 0.9818
    Epoch 28/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0583 - accuracy: 0.9791 - val_loss: 0.0496 - val_accuracy: 0.9825
    Epoch 29/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0567 - accuracy: 0.9788 - val_loss: 0.0482 - val_accuracy: 0.9829
    Epoch 30/30
    180/180 [==============================] - 2s 10ms/step - loss: 0.0556 - accuracy: 0.9802 - val_loss: 0.0478 - val_accuracy: 0.9827



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f630bcf9650>




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
    


    180/180 [==============================] - 4s 17ms/step - loss: 0.6708 - accuracy: 0.6588 - val_loss: 0.6150 - val_accuracy: 0.9261
    Epoch 2/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.4990 - accuracy: 0.9042 - val_loss: 0.3762 - val_accuracy: 0.9393
    Epoch 3/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.3160 - accuracy: 0.9295 - val_loss: 0.2496 - val_accuracy: 0.9528
    Epoch 4/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.2321 - accuracy: 0.9456 - val_loss: 0.1926 - val_accuracy: 0.9602
    Epoch 5/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1908 - accuracy: 0.9533 - val_loss: 0.1615 - val_accuracy: 0.9652
    Epoch 6/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1640 - accuracy: 0.9582 - val_loss: 0.1412 - val_accuracy: 0.9681
    Epoch 7/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1451 - accuracy: 0.9642 - val_loss: 0.1265 - val_accuracy: 0.9697
    Epoch 8/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1318 - accuracy: 0.9668 - val_loss: 0.1161 - val_accuracy: 0.9715
    Epoch 9/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1190 - accuracy: 0.9706 - val_loss: 0.1081 - val_accuracy: 0.9728
    Epoch 10/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1099 - accuracy: 0.9708 - val_loss: 0.1021 - val_accuracy: 0.9755
    Epoch 11/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.1035 - accuracy: 0.9723 - val_loss: 0.0984 - val_accuracy: 0.9771
    Epoch 12/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0965 - accuracy: 0.9739 - val_loss: 0.0930 - val_accuracy: 0.9773
    Epoch 13/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0903 - accuracy: 0.9753 - val_loss: 0.0933 - val_accuracy: 0.9764
    Epoch 14/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0843 - accuracy: 0.9764 - val_loss: 0.0881 - val_accuracy: 0.9789
    Epoch 15/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0797 - accuracy: 0.9779 - val_loss: 0.0862 - val_accuracy: 0.9786
    Epoch 16/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0776 - accuracy: 0.9791 - val_loss: 0.0838 - val_accuracy: 0.9795
    Epoch 17/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0726 - accuracy: 0.9806 - val_loss: 0.0821 - val_accuracy: 0.9798
    Epoch 18/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0691 - accuracy: 0.9811 - val_loss: 0.0812 - val_accuracy: 0.9795
    Epoch 19/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0671 - accuracy: 0.9813 - val_loss: 0.0807 - val_accuracy: 0.9793
    Epoch 20/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0620 - accuracy: 0.9826 - val_loss: 0.0798 - val_accuracy: 0.9795
    Epoch 21/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0615 - accuracy: 0.9818 - val_loss: 0.0816 - val_accuracy: 0.9793
    Epoch 22/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0597 - accuracy: 0.9826 - val_loss: 0.0786 - val_accuracy: 0.9802
    Epoch 23/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0549 - accuracy: 0.9848 - val_loss: 0.0793 - val_accuracy: 0.9802
    Epoch 24/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0541 - accuracy: 0.9844 - val_loss: 0.0808 - val_accuracy: 0.9800
    Epoch 25/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0510 - accuracy: 0.9851 - val_loss: 0.0785 - val_accuracy: 0.9802
    Epoch 26/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0501 - accuracy: 0.9857 - val_loss: 0.0815 - val_accuracy: 0.9804
    Epoch 27/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0482 - accuracy: 0.9851 - val_loss: 0.0799 - val_accuracy: 0.9809
    Epoch 28/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0431 - accuracy: 0.9862 - val_loss: 0.0787 - val_accuracy: 0.9807
    Epoch 29/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0463 - accuracy: 0.9849 - val_loss: 0.0781 - val_accuracy: 0.9811
    Epoch 30/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0429 - accuracy: 0.9873 - val_loss: 0.0823 - val_accuracy: 0.9818



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f63001e54d0>




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
    180/180 [==============================] - 5s 23ms/step - loss: 0.6202 - accuracy: 0.6962 - val_loss: 0.3579 - val_accuracy: 0.9371
    Epoch 2/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.2271 - accuracy: 0.9373 - val_loss: 0.1470 - val_accuracy: 0.9640
    Epoch 3/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.1395 - accuracy: 0.9608 - val_loss: 0.1100 - val_accuracy: 0.9715
    Epoch 4/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.1089 - accuracy: 0.9688 - val_loss: 0.0914 - val_accuracy: 0.9755
    Epoch 5/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0873 - accuracy: 0.9764 - val_loss: 0.0761 - val_accuracy: 0.9802
    Epoch 6/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0666 - accuracy: 0.9814 - val_loss: 0.0565 - val_accuracy: 0.9843
    Epoch 7/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0434 - accuracy: 0.9883 - val_loss: 0.0423 - val_accuracy: 0.9874
    Epoch 8/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0336 - accuracy: 0.9901 - val_loss: 0.0313 - val_accuracy: 0.9915
    Epoch 9/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0275 - accuracy: 0.9923 - val_loss: 0.0274 - val_accuracy: 0.9939
    Epoch 10/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0225 - accuracy: 0.9936 - val_loss: 0.0253 - val_accuracy: 0.9942
    Epoch 11/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0197 - accuracy: 0.9944 - val_loss: 0.0231 - val_accuracy: 0.9948
    Epoch 12/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0190 - accuracy: 0.9940 - val_loss: 0.0219 - val_accuracy: 0.9946
    Epoch 13/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0154 - accuracy: 0.9957 - val_loss: 0.0216 - val_accuracy: 0.9946
    Epoch 14/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0146 - accuracy: 0.9954 - val_loss: 0.0206 - val_accuracy: 0.9951
    Epoch 15/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0128 - accuracy: 0.9961 - val_loss: 0.0194 - val_accuracy: 0.9953
    Epoch 16/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0126 - accuracy: 0.9965 - val_loss: 0.0194 - val_accuracy: 0.9953
    Epoch 17/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0109 - accuracy: 0.9967 - val_loss: 0.0188 - val_accuracy: 0.9951
    Epoch 18/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0127 - accuracy: 0.9958 - val_loss: 0.0179 - val_accuracy: 0.9951
    Epoch 19/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0086 - accuracy: 0.9974 - val_loss: 0.0206 - val_accuracy: 0.9935
    Epoch 20/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0074 - accuracy: 0.9974 - val_loss: 0.0183 - val_accuracy: 0.9953
    Epoch 21/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0093 - accuracy: 0.9968 - val_loss: 0.0233 - val_accuracy: 0.9915
    Epoch 22/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0089 - accuracy: 0.9968 - val_loss: 0.0182 - val_accuracy: 0.9951
    Epoch 23/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0087 - accuracy: 0.9972 - val_loss: 0.0194 - val_accuracy: 0.9948
    Epoch 24/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0086 - accuracy: 0.9972 - val_loss: 0.0188 - val_accuracy: 0.9953
    Epoch 25/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0084 - accuracy: 0.9971 - val_loss: 0.0197 - val_accuracy: 0.9951
    Epoch 26/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0070 - accuracy: 0.9977 - val_loss: 0.0187 - val_accuracy: 0.9951
    Epoch 27/30
    180/180 [==============================] - 4s 21ms/step - loss: 0.0086 - accuracy: 0.9968 - val_loss: 0.0231 - val_accuracy: 0.9948
    Epoch 28/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0077 - accuracy: 0.9977 - val_loss: 0.0191 - val_accuracy: 0.9953
    Epoch 29/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0071 - accuracy: 0.9979 - val_loss: 0.0178 - val_accuracy: 0.9955
    Epoch 30/30
    180/180 [==============================] - 4s 22ms/step - loss: 0.0057 - accuracy: 0.9981 - val_loss: 0.0184 - val_accuracy: 0.9951



```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f628c3c4890>




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

    225/225 [==============================] - 3s 12ms/step - loss: 0.0312 - accuracy: 0.9930





    [0.031193897128105164, 0.9930063486099243]



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
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

fig.show()
# Ready to plot!
```


This embedding seems to have learned some reasonable associations.

For example, we see that words like "London", "Spain", 'China','Japan','turkey', and "capital" are relatively close to each other. So are "to", "for", and "of", as well as "century", "21st", and "daily."
