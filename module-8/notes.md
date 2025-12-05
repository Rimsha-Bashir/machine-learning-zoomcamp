## 8.1 Fashion Classification 

Learning about Neural Network and Deep Learning. 

Example - Fashion Clasification 

This is a multi-class classification where the IMG can be of several different types. In the example were taking, classifying fashion - classes could be T-Shirts, Trousers, Shorts, Skirts etc. 

We'll use a neural net model to accomplish this. 

## 8.2 TensorFlow & Keras

Tensorflow is a framework/open-source library for deep learning created by Google. 

Use cases:
- Recognizing Images 
- Understanding Speech 
- Translating languages
- Making predictions (like stock prices or trends)

Keras is a high-level API built on top of tensorflow - it provides higher level obstruction, so you don't have to write long, complex code. 

TensorFlow does the heavy lifting in the background,
and Keras gives you the friendlier interface to build your models.

```python
from tensorflow import keras
```

What is a GPU (for machine learning)?

A GPU (Graphics Processing Unit) performs thousands of operations in parallel, making it much faster than a CPU for training neural networks.

Can I use a GPU in Jupyter Notebook?

Yes, if your environment provides one.

1. Using Jupyter in the Cloud

Google Colab or Kaggle Notebooks

Free GPUs available.

In Colab:
Runtime → Change runtime type → Hardware accelerator → GPU

Test with:

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

2. Using Jupyter on Your Own Computer

You can use GPU acceleration if you have:

An NVIDIA GPU

NVIDIA drivers

CUDA installed

cuDNN installed

TensorFlow installed:

    `pip install tensorflow`


TensorFlow will automatically detect the GPU if configured correctly.

3. If You Do Not Have a GPU

You can still run TensorFlow, but training will be slower.
Cloud notebooks are the easiest way to use a free GPU.

Quick GPU Detection:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

If it lists a GPU → GPU is active

If it returns [] → No GPU available

## 8.3 Pre-trained convolution neural networks

[Check nb-1](./fashion-classification.ipynb)
and 
[Check nb-2](https://colab.research.google.com/drive/1BQvZEn11J9XUZj8TqM2zA3gv22kyIahD?usp=sharing)


## 8.4 Convolutional Neural Networks 

[Check nb-2](https://colab.research.google.com/drive/1BQvZEn11J9XUZj8TqM2zA3gv22kyIahD?usp=sharing)

A CNN has two types of layers - convolutional, and dense. 

![alt text](./images/8.4.1.png)

Convolutional layers consists of `filters` which are like small images (eg. 5x5 sized etc.). It contains simple shapes as seen in the img above. 

and a CNN model runs the filter over the image and outputs a number signifying the likeness/similarity of the filter to the particular part of the image

the resulting matrix is a `feature map`

**there will me as many feature maps as there are filters**

Hence, the output of a convolution layer (CNL1) is the feature map - which can be our new image. 

The next convolution layer has it's own set of filters and produces a new feature map. 

Let's say we have 6 set of features, and so, we'll have 6 feature maps. 

![alt text](./images/8.4.2.png)

feature set 2 will be more complex than feature set 1 - combination of filters for the first set will look like one from set 2

for ex. (as in the image) 2 slanted lines in opp directions when put together form an `X`


<details>
<summary>NOTES SUMMARY</summary>

1. Filters and Feature Maps

- **Filters** (also called **kernels**) are small matrices, e.g., 3×3 or 5×5, that detect specific patterns in the image (like vertical lines, horizontal lines, or edges). ✅ Correct
- When you **slide a filter over the image** (convolution), it produces a number at each position — this is the **activation**, showing how much that pattern matches that part of the image. ✅ Correct
- Collecting all these numbers gives the **feature map**. ✅ Correct
- You have **one feature map per filter**. ✅ Correct
- Multiple filters can detect multiple patterns in the same image → multiple feature maps per layer. ✅ Correct

Your description of combining simple features (lines) into more complex ones (like an “X”) is also correct — deeper layers capture more complex patterns. ✅

2. Two Layers vs Multiple Convolutional Layers

- Saying a CNN has “two layers: convolutional and dense” usually refers to **types of layers**, not the number of convolutional layers.
- A real CNN usually has **many convolutional layers**, each with its own set of filters.
- First convolutional layers detect **basic patterns** (edges, lines).
- Later convolutional layers detect **more complex patterns** (shapes, corners, objects) by combining feature maps from previous layers.

✅ **“Two types of layers” ≠ only two convolutional layers**

3. Filter vs Filter Set

- Each **filter** detects **one pattern**.
- A **set of filters** in a layer is the collection of all filters in that layer.
- Example:
  - Layer 1: 6 filters → 6 feature maps
  - Layer 2: 6 filters → 6 new feature maps


4. Feature Map as an Image

- A **feature map** is a 2D matrix of numbers. Each number represents how strongly a filter activated at that location.
- You can **visualize it as a grayscale image**, where brighter pixels mean stronger activation.
- After the first convolutional layer, the output (all feature maps) can be treated as a **new image with multiple channels**, which becomes input to the next convolutional layer. ✅

**Example:**

- Input image: 28×28×1 (grayscale)
- Layer 1: 6 filters → 6 feature maps → shape 28×28×6
- Layer 2: 6 filters → each sees all 6 feature maps from Layer 1 → 6 new feature maps → shape 28×28×6 (or smaller if pooling applied)

5. Summary of Key Questions

| Question | Answer |
| --- | --- |
| Two layers vs multiple CNN layers | “Two layers” usually means **types of layers** (convolutional + dense). A CNN can have multiple convolutional layers. |
| Filter set or filter | Each **filter** detects one pattern. The **set of filters** in a layer produces multiple feature maps. |
| Feature map count | One feature map per filter. So 6 filters → 6 feature maps. Each new convolutional layer has its own set of filters → new feature maps. |
| Feature map as an image | A feature map is a 2D matrix of activations. You can **visualize it as an image** to see what the filter is detecting. |


6. Key Clarifications

- “Two layers” = **layer types**, not the literal count of convolutional layers.
- Each convolutional layer usually has multiple filters → multiple feature maps.
- Feature maps can be treated as images → input for the next convolutional layer.
</details>


The output of applying the CONV layers is `vector representation`. 
Now, with this vector rep we create dense layers that give us the final output/prediction, which, here, is say, a t-shirt. 

**For binary classification:**

We take the xi to xn in the vector and multiply it by weights w1 to wn and sum them, then apply the sigmoid fnc (here it will be, t-shirt, or no t-shirt)

g(x) = sigmoid(x<sup>T</sup>w)

![alt text](./images/8.4.3.png)

**For multiclass classification:**

![alt text](./images/8.4.4.png)

Here, sigmoid is replaced with `soft max` - generalized fnc, and w is what differs for each item. 

Basically, multiple logistic regression models put together. 

![alt text](./images/8.4.5.png)

The space/layer between the input and output is what we call the `dense layer` -> because of all the connections between elements of input and outputs. 

Pooling is what makes a feature map smaller. eg. if your feature map is 200x200, then after pooling it has 100x100. 

For an in-depth understanding of how CNN's work for images, check this https://cs231n.github.io/convolutional-networks/


## 8.5 Transfer Learning

Transfer learning is when you take a model (Model A) that has already been trained on a large, general dataset, and reuse part or all of it to help train a new model (Model B) on a smaller or more specific dataset.

Here, we're using the trained layers of convolutional networks for the dense layers.  

input -> conv layers -> vector rep -> dense layers -> prediction

`conv layers + vector rep` are already trained on imagenet and they're more or less generic so we don't need to change them. 

dense layers are specific to the dataset (we disuse what we get from imagenet)

so we keep conv layers and train dense layers. 

....

check gcolab notes 

![alt text](./images/8.5.1.png)


image -> conv layers -> vector rep

our base model here is Xception. 

we will train the rest (dense layers/custom model part)

![alt text](./images/8.5.2.png)

Keras uses a bottom up approach and refers to dense layers + output as the top layer and convolutional netw + vector rep + input as the bottom layer

thus, when you create a base_model 

i.e 

base_model = Xception(weights='imagenet', include_top=False, input_shape(150,150,3))

we don't want to use this base model, we only want to extract vector rep from it. 

.... check gcolab

![alt text](./images/8.5.3.png)

The above is what we call pooling - reducing the dimensionality of something so it can be represented in 1D

.... check gcolab

## 8.6 Adjusting the learning rate

Learning rate is how fast you read/learn. 

If you read 10+ books p/year, you're learning rate is high. 

If you read 4 p/year, you're learning rate is medium 

If you read 1 p/year, you're learning rate is slow. 

It's more likely you learnt everything very well/the best in the last case, but problem is that it takes too long. In case 1, maybe you forget and don't retain info as well, which means our model will do poorly in validation and cause overfitting. 

In case 3, you underfit, because it was possible to learn faster. 

So what we're looking to do is here, is find an appropriate learning rate (no. of books, say) that doesn't comprosomise on performance quality while also taking less time. 




## 8.7 Checkpointing

Way of saving the model after each iteration or after certain conditions are met. 

In a Convolutional Neural Network (CNN), an epoch is a single pass through the entire training dataset.

![alt text](./images/8.5.4.png)

save_best_only: saves only the best accuracy epochs. 

keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True, 
    monitor='val_accuracy',
    mode='max' # because we want to maximize accuracy, if we were monitoring losses, then mdode=min
    )

  The above is how you create a checkpoint. 