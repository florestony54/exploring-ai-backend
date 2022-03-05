---
layout: post
current: post
cover:  assets/images/a2.jpg
navigation: True
title: Implementing Batch Normalization with Keras API
date: 2022-02-19 23:00:00
tags: [deep learning]
class: post-template
subclass: 'post'
author: tony
---

The following is an excerpt from the book "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurelien Geron (temporary filler content):

### Implementing Batch Normalization with Keras
As with most things with Keras, implementing Batch Normalization is quite simple.
Just add a BatchNormalization layer before or after each hidden layer’s activation
function, and optionally add a BN layer as well as the first layer in your model. For
example, this model applies BN after every hidden layer and as the first layer in the
model (after flattening the input images):</p>

```
 model = keras.models.Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.BatchNormalization(),
 keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
 keras.layers.BatchNormalization(),
 keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
 keras.layers.BatchNormalization(),
 keras.layers.Dense(10, activation="softmax")
])
```

That’s all! In this tiny example with just two hidden layers, it’s unlikely that Batch
Normalization will have a very positive impact, but for deeper networks it can make a
tremendous difference.
Let’s zoom in a bit. If you display the model summary, you can see that each BN layer
adds 4 parameters per input: γ, β, μ and σ (for example, the first BN layer adds 3136
parameters, which is 4 times 784). The last two parameters, μ and σ, are the moving
averages, they are not affected by backpropagation, so Keras calls them “Nontrainable”9
 (if you count the total number of BN parameters, 3136 + 1200 + 400, and
divide by two, you get 2,368, which is the total number of non-trainable params in
this model).

```
>>> model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
flatten_3 (Flatten) (None, 784) 0
_________________________________________________________________
batch_normalization_v2 (Batc (None, 784) 3136
_________________________________________________________________
dense_50 (Dense) (None, 300) 235500
_________________________________________________________________
batch_normalization_v2_1 (Ba (None, 300) 1200
_________________________________________________________________
dense_51 (Dense) (None, 100) 30100
_________________________________________________________________
batch_normalization_v2_2 (Ba (None, 100) 400
_________________________________________________________________
dense_52 (Dense) (None, 10) 1010
=================================================================
Total params: 271,346
Trainable params: 268,978
Non-trainable params: 2,368 
```

Let’s look at the parameters of the first BN layer. Two are trainable (by backprop), and
two are not:
```
>>> [(var.name, var.trainable) for var in model.layers[1].variables]
[('batch_normalization_v2/gamma:0', True),
 ('batch_normalization_v2/beta:0', True),
 ('batch_normalization_v2/moving_mean:0', False),
 ('batch_normalization_v2/moving_variance:0', False)]
```

Now when you create a BN layer in Keras, it also creates two operations that will be
called by Keras at each iteration during training. These operations will update the
336 | Chapter 11: Training Deep Neural Networks
moving averages. Since we are using the TensorFlow backend, these operations are
TensorFlow operations (we will discuss TF operations in Chapter 12).

```
>>> model.layers[1].updates
[<tf.Operation 'cond_2/Identity' type=Identity>,
 <tf.Operation 'cond_3/Identity' type=Identity>]
```

The authors of the BN paper argued in favor of adding the BN layers before the acti‐
vation functions, rather than after (as we just did). There is some debate about this, as
it seems to depend on the task. So that’s one more thing you can experiment with to
see which option works best on your dataset. To add the BN layers before the activa‐
tion functions, we must remove the activation function from the hidden layers, and
add them as separate layers after the BN layers. Moreover, since a Batch Normaliza‐
tion layer includes one offset parameter per input, you can remove the bias term from
the previous layer (just pass use_bias=False when creating it):

```
model = keras.models.Sequential([
 keras.layers.Flatten(input_shape=[28, 28]),
 keras.layers.BatchNormalization(),
 keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
 keras.layers.BatchNormalization(),
 keras.layers.Activation("elu"),
 keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
 keras.layers.Activation("elu"),
 keras.layers.BatchNormalization(),
 keras.layers.Dense(10, activation="softmax")
])
```

The BatchNormalization class has quite a few hyperparameters you can tweak. The
defaults will usually be fine, but you may occasionally need to tweak the momentum.
This hyperparameter is used when updating the exponential moving averages: given a
new value v (i.e., a new vector of input means or standard deviations computed over
the current batch), the running average ᅲ is updated using the following equation:

> v <- v × momentum + v × 1 − momentum

A good momentum value is typically close to 1—for example, 0.9, 0.99, or 0.999 (you
want more 9s for larger datasets and smaller mini-batches).
Another important hyperparameter is axis: it determines which axis should be nor‐
malized. It defaults to –1, meaning that by default it will normalize the last axis (using
the means and standard deviations computed across the other axes). For example,
when the input batch is 2D (i.e., the batch shape is [batch size, features]), this means
that each input feature will be normalized based on the mean and standard deviation
computed across all the instances in the batch. For example, the first BN layer in the
previous code example will independently normalize (and rescale and shift) each of
the 784 input features. However, if we move the first BN layer before the Flatten
layer, then the input batches will be 3D, with shape [batch size, height, width], there‐
fore the BN layer will compute 28 means and 28 standard deviations (one per column
of pixels, computed across all instances in the batch, and all rows in the column), and
it will normalize all pixels in a given column using the same mean and standard devi‐
ation. There will also be just 28 scale parameters and 28 shift parameters. If instead
you still want to treat each of the 784 pixels independently, then you should set
axis=[1, 2].
Notice that the BN layer does not perform the same computation during training and
after training: it uses batch statistics during training, and the “final” statistics after
training (i.e., the final value of the moving averages). Let’s take a peek at the source
code of this class to see how this is handled:
```
class BatchNormalization(Layer):
 [...]
 def call(self, inputs, training=None):
 if training is None:
 training = keras.backend.learning_phase()
 [...]
 ```
The call() method is the one that actually performs the computations, and as you
can see it has an extra training argument: if it is None it falls back to keras.back
end.learning_phase(), which returns 1 during training (the fit() method ensures
that). Otherwise, it returns 0. If you ever need to write a custom layer, and it needs to
behave differently during training and testing, simply use the same pattern (we will
discuss custom layers in Chapter 12).
Batch Normalization has become one of the most used layers in deep neural net‐
works, to the point that it is often omitted in the diagrams, as it is assumed that BN is
added after every layer. However, a very recent paper10 by Hongyi Zhang et al. may
well change this: the authors show that by using a novel fixed-update (fixup) weight
initialization technique, they manage to train a very deep neural network (10,000 lay‐
ers!) without BN, achieving state-of-the-art performance on complex image classifi‐
cation tasks