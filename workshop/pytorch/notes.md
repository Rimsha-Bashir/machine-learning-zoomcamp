## Learning PyTorch


[YT Link](https://www.youtube.com/watch?v=r1bquDz5GGA)

`Tensor` - torch.Tensor is the basic building block of PyTorch. 

### Three ways/patterns to create a tensor:

1. Direct creation from data:

![alt text](./images/1.png)

2. Creation from a desired shape:

Commonly used when initializing model weights. 

![alt text](./images/2.png)

3. Creation by mimicking another tensor:

![alt text](./images/3.png)


### What's inside a tensor?

A tensor has three critical attributes - shape, datatype and device. 

.shape: a tuple describing dimensions, ex. (2,3) means 2 rows and 3 columns. (:warning: 90% of errors in pytorch will be because of type mismatches)

.device: tells us where the tensor lives, i.e, in CPU or CUDA(GPU)

.dtype: the data type of numbers in the tensor, the default is float. It is float because of gradients. The entire ML engine runs on weights/biases adjustments (ex. 3.000 to 3.001). Whatever `learns` has to be a float32. 

What is Autograd?

Autograd : Automatic differentiation. 
It's PyTorch's built in gradient calculator and it can be activated by the below parameter:

`requires_grad=True`

By default, a tensor is just data, to tell PyTorch that it's a learnable parameter, you must set `requires_grad=True`. By doing so, you're sending a message to the autograd engine, "This is a parameter. From now on, track every single operation that happens to it"

![alt text](./images/4.png)

Note that by defailt, requires_grad is False.

As soon as the parameter is set, PyTorch begins to build a computation graph 

Let's say we compute z=x.y, where y=a+b 

![alt text](./images/5.png)


### Difference between TensorFlow and PyTorch 

| Aspect              | PyTorch   | TensorFlow |
| ------------------- | --------- | ---------- |
| Learning curve      | Easier    | Steeper    |
| Debugging           | Very easy | Harder     |
| Research popularity | Very high | Moderate   |
| Production tools    | Good      | Excellent  |
| Mobile & edge       | Limited   | Strong     |
| Flexibility         | High      | Medium     |
| Performance         | Excellent | Excellent  |

<details><summary>details here^</summary>

## 2. Shared Core Ideas (TensorFlow & PyTorch)

Both frameworks:

- Use **tensors** as the main data structure  
- Support **GPU / TPU acceleration**  
- Automatically compute **gradients** (backpropagation)  
- Can train **deep neural networks**  
- Are used in **research and production**

So the difference is **not what they can do**, but **how they feel and where they shine**.

---

## 3. The Fundamental Difference (Mental Model)

### PyTorch: *“Define-by-Run” (Dynamic)*

- Code runs **line by line**
- The computation graph is built **as the code executes**
- Feels like **normal Python**

### TensorFlow: *“Define-then-Run” (Historically Static)*

- You define a **computation graph** first
- Then **execute** it
- More **structured** and optimized for **deployment**

> Modern TensorFlow (with eager execution) feels more like PyTorch, but the **design philosophy difference still exists**.

---

## 4. PyTorch Explained

### What PyTorch Feels Like

- Very **Pythonic**
- **Easy to debug**
- **Flexible**

### Why People Choose PyTorch

- Research-friendly
- Rapid experimentation
- Easy to understand model behavior

### Typical Use Cases

- Academic research
- Prototyping new models
- Custom architectures
- Computer vision research
- NLP research

### Example Thought

> *“I want to try something weird with my model during training.”*

**PyTorch says:**  
👉 *No problem, just write Python code.*

---

## 5. TensorFlow Explained

### What TensorFlow Feels Like

- More **structured**
- More tools **out-of-the-box**
- Strong **deployment ecosystem**

### Why People Choose TensorFlow

- Production deployment
- Mobile and embedded devices
- Large-scale systems

### Typical Use Cases

- Production ML systems
- Mobile ML (**TensorFlow Lite**)
- Web ML (**TensorFlow.js**)
- Large enterprise pipelines
- Serving models at scale

### Example Thought

> *“I want to train a model and deploy it everywhere.”*

**TensorFlow says:**  
👉 *Here’s a complete ecosystem.*
</details>