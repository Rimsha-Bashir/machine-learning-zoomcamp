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