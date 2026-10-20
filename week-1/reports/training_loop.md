# Anatomy of the Neural Network Training Loop

A neural network learns by repeatedly making predictions, measuring its errors, calculating how its parameters contributed to those errors, and updating those parameters to improve future predictions. This process is called the **training loop**.

## 1. Forward Pass

During the forward pass, input images are passed through the network layer by layer. Each layer performs mathematical operations using its weights and biases. In a CNN, convolutional layers extract visual features, activation functions introduce non-linearity, and the final layer produces scores called *logits* for each class.

For example, when a CIFAR-10 image is passed through the model, the network outputs ten logits corresponding to classes such as airplane, cat, and automobile.

```python
logits = model(images)
```

The forward pass produces predictions but does not update the model's parameters.

## 2. Loss Calculation

The loss function measures how well the predictions match the correct labels. For multi-class classification, PyTorch's `CrossEntropyLoss` is commonly used. It compares the predicted class scores with the actual labels and produces a numerical error value.

```python
loss = criterion(logits, labels)
```

A lower loss generally indicates better predictions. However, loss and accuracy are different metrics: loss measures prediction error, while accuracy measures the proportion of correctly classified images.

## 3. Backpropagation

Backpropagation calculates how the loss changes with respect to each trainable parameter using the chain rule of calculus. PyTorch's autograd engine tracks the operations performed during the forward pass and uses this computational graph to calculate gradients.

```python
optimizer.zero_grad()
loss.backward()
```

The first line clears gradients from the previous batch because PyTorch accumulates gradients by default. The second calculates gradients for the current batch. These gradients indicate how changing the weights and biases would affect the loss.

## 4. Optimizer Step

After gradients have been calculated, the optimizer uses them to update the model's parameters. For basic gradient descent, the update rule is:

\[
\theta_{\text{new}} = \theta_{\text{old}} -
\eta \nabla_{\theta} L
\]

Here, \(\theta\) represents the parameters, \(\eta\) is the learning rate, and \(\nabla_{\theta} L\) represents the loss gradients.

```python
optimizer.step()
```

The learning rate controls the size of parameter updates. If it is too large, training may become unstable; if it is too small, learning may be slow. Optimizers such as Adam adapt updates using additional information from previous gradients.

## 5. Why Normalization Helps

Image pixels are often represented by values between 0 and 255. Scaling them to between 0 and 1, or standardizing each channel using its mean and standard deviation, puts the input values on a more suitable numerical scale.

For example:

\[
x_{\text{normalized}} = \frac{x-\mu}{\sigma}
\]

Normalization can improve optimization stability and help the model learn more effectively. In this project, CIFAR-10 images are normalized using channel-specific statistics. The same statistics must be applied consistently to training, validation, and test images.

**Batch normalization** is a separate technique used inside the network to normalize intermediate activations. It can stabilize training and often helps deeper networks optimize effectively.

## Summary

The training loop follows this sequence:

**Forward pass → Loss calculation → Zero gradients → Backpropagation → Optimizer step**

This sequence repeats for each batch across multiple epochs. Over time, the optimizer adjusts the network's parameters to reduce prediction errors. Validation data is used to monitor generalization and select a good model, while the test set is reserved for the final performance evaluation.
