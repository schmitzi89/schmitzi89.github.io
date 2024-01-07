---
layout: post
title:  "Deep Learning: Why Object-Oriented?"
---

## Introduction

Why are deep learning frameworks implemented with object-oriented programming? It is a question that programmers probably don`t ask themselves but I have a maths and R background where people typically use functional programming. I asked myself this question when I looked at the way layers and models are implemented in Keras.

For example, this is a very naive object-oriented implementation of a dense layer:

```R
layer_naive_dense <- function(input_size, output_size, activation) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"

  self$activation <- activation

  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)

  b_shape <- c(output_size)
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)

  self$weights <- list(self$W, self$b)

  self$call <- function(inputs) {
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }
  self
}
```

Here, we are creating a function `layer_naive_dense` that creates an object of type `NaiveDense` when initiated. The object will have the following properties:

- W: The weights of the layer.
- b: The biases of the layer.
- weights: A list that contains both weights and biases.
- activation: The activation function to be used in the layer.

Furthermore, it has a method `call` which will do the matrix multiplications and return the transformed input. The most important aspect to notice is that an object contains a local environment which means that methods and properties live in the environment of the object.

But why don´t we just implement it as a simple function without local environments? Because from a mathematical perspective, a dense layer is just a simple function. We can manage everything that is important for the function outside of it and pass it as a variables . The following would be a simple `layer_naive_dense` function which does just that:

```R
layer_naive_dense <- function(inputs, W, b, activation) {
  activation(tf$matmul(inputs, W) + b)
}
```

We save a lot of code! But we would now have to manage the weights outside of the function. That means the weights need to be initialized outside of the dense layer and passed as variables to the function. Would that make the code easier or more difficult overall?

Deep Learning models are just simple functions - chained. Everything needs to be differentiable to make backpropagation work. The essential machine-learning steps are simple functions appied one after the other. Would it help us, if we implement deep-learning in functional programming style? Would it make the code more explicit and easy to understand? Lets try it out and write an end-to-end deep learning implementation to train a model that classifies MNIST (handwritten number classification) ! I will try to keep everything low-level so we can really compare piece by piece.

## The object-oriented implementation

Our object-oriented implementation will be the following. It is largely taken from _Deep Learning with R (second edition)_ from François Chollet.

```R
library(keras)
library(tensorflow)

random_array <- function(dim, min = 0, max = 1){
  array(runif(prod(dim), min, max),dim)
}

layer_naive_dense <- function(input_size, output_size, activation) {

  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"

  self$activation <- activation

  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)

  b_shape <- c(output_size)
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)

  self$weights <- list(self$W, self$b)

  self$call <- function(inputs) {
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }
  self
}

naive_model_sequential <- function(layers) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"

  self$layers <- layers

  weights <- lapply(layers, function(layer) layer$weights)
  self$weights <- do.call(c, weights)

  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }
  self
}

new_batch_generator <- function(images, labels, batch_size = 128) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "BatchGenerator"

  stopifnot(nrow(images) == nrow(labels))
  self$index <- 1
  self$images <- images
  self$labels <- labels
  self$batch_size <- batch_size
  self$num_batches <- ceiling(nrow(images) / batch_size)

  self$get_next_batch <- function() {
    start <- self$index
    if(start > nrow(images))
      return(NULL)

    end <- start + self$batch_size - 1
    if(end > nrow(images))
      end <- nrow(images)

    self$index <- end + 1
    indices <- start:end
    list(images = self$images[indices, ],
         labels = self$labels[indices])
  }
  self
}

one_epoch <- function(model, images_batch, labels_batch) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model$call(images_batch)
    per_sample_losses <-
      loss_sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss <- mean(per_sample_losses)
  })
  gradients <- tape$gradient(average_loss, model$weights)
  update_weights(gradients, model$weights)
  average_loss
}

update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub(gradients[[i]] * 1e-3)
}

fit_model <- function(model, images, labels, epochs, batch_size = 128, test_images, test_labels) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels)
    for (batch_counter in seq_len(batch_generator$num_batches)) {
      batch <- batch_generator$get_next_batch()
      loss <- one_training_step(model, batch$images, batch$labels)
    }
    predictions <- model$call(test_images)
    predicted_labels <- max.col(predictions) - 1
    matches <- predicted_labels == test_labels
    cat(sprintf("accuracy: %.2f\n", mean(matches)))
  }
}
```

After we load the functions above, we can train a model to classify MNIST with the following code:

```R
mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28)) / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28)) / 255
test_labels <- mnist$test$y
train_labels <- mnist$train$y

model <- naive_model_sequential(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512, activation = tf$nn$relu),
  layer_naive_dense(input_size = 512, output_size = 10, activation = tf$nn$softmax)
))

fit_model(model, train_images, train_labels, epochs = 1, batch_size = 128, test_images, test_labels)
```

After ~30 Epochs it reaches an accuracy of around 90% which sounds great but is actually really bad because if we replace our simple weights update with a standard optimizer like rmsprop it would reach ~98% accuracy after 5 epochs.
Anyway, this post is not about training the best model. It is about understanding the differences between object-oriented and functional programming implementations.

We can generate predictions with:

```R
predictions <- model$call(test_images)
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == test_labels
cat(sprintf("accuracy: %.2f\n", mean(matches)))
```

Easy to use, right?

How would that look like if we implement it via functional programming?

## Step by step guide to the functional implementation

In order to build the functional implementation step by step, lets look at the individual parts of the object-oriented implementation from above and try to implement everything using functional programming paradigms:

- We wont use local environments
- We want to explicitly pass all required parameters to the function
- The functions should return whats needed for all parts to work well together

### Generator function

I am starting with the generator function since this is a very easy example. I will write down the implementation and discuss the differences of the functional implementation vs the object-oriented implementation.

#### The OO-Implementation

```R
new_batch_generator <- function(images, labels, batch_size = 128) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "BatchGenerator"

  stopifnot(nrow(images) == nrow(labels))
  self$index <- 1
  self$images <- images
  self$labels <- labels
  self$batch_size <- batch_size
  self$num_batches <- ceiling(nrow(images) / batch_size)

  self$get_next_batch <- function() {
    start <- self$index
    if(start > nrow(images))
      return(NULL)

    end <- start + self$batch_size - 1
    if(end > nrow(images))
      end <- nrow(images)

    self$index <- end + 1
    indices <- start:end
    list(images = self$images[indices, ],
         labels = self$labels[indices])
  }
  self
}
```

Afterwards, we call the OO-function like this:

```R
batch_generator <- new_batch_generator(images, labels)

for (batch_counter in seq_len(batch_generator$num_batches)) {
  batch <- batch_generator$get_next_batch()
  ...
}
```

#### The Functional implementation

```R
new_batch_generator <- function(images, labels, batch_size = 128, batch_counter) {
  start <- batch_size * (batch_id -1) + 1
  end <- batch_size * batch_id
  if(end >= nrow(images)) {
    end <- nrow(images)
  }
  indices <- start:end
  return(list(images = images[indices, ], labels = labels[indices]))
}
```

Afterwards, we call the function like this:

```R
num_batches <- ceiling(nrow(images) / batch_size)

for(batch_counter in seq_len(num_batches)){
  batch <- new_batch_generator(images, labels, batch_counter)
  ...
}
```

#### Differences

| Aspect | Conclusion | Winner | 
|----------|----------|----------|
| Clarity of code | Object-oriented style is more verbose. Object properties need to be defined inside the function. | Functional |
| Ease of use | The OO batch generator is slightly easier to use. It just needs to be instantiated. In the functional implementation, the number of batches need to be calculated and the batch counter needs to be passed. | OO |
| Flexibility | In the OO implementation it is more difficult if we want to get a specific batch, say the fourth batch, we would need to do: <pre>batch_generator <- new_batch_generator(images, labels)<br>batch_generator$index <- batch_size*3 +1 # we need to set the index manually to the end of the third batch<br>batch_generator$get_next_batch()</pre> In the functional implementation it can be done like this: <pre>new_batch_generator(images, labels, batch_counter = 4)</pre> Which is much more straightforward. | Functional |
| Debugability | In the OO implementation, we can quickly extract meta data regarding the batch generator from the batch generator object: batch size, index, total number of batches. In the functional implementation this info is available in the current environment. | Tie |

#### Conclusion

For the generator function, my winner is the functional implementation. Having to manage the batch count outside of the generator and having to pass it to the function in the functional implementation is a small disadvantage which comes with greater benefits like the much clearer implementation and greater flexibility.
However, the user needs to know how to embed the function into a higher-level code. With the OO implementation he does not need to worry how the number of batches should be calculated since the object will take care of it itself. That aspect makes the object-oriented implementation more encapsulated. Bugs might easier happen with the functional implementation, when users do not correctly feed the required batch counter.
In the case of the generator function it is a minor issue since the logic is so easy but lets see how things turn out for the other functions of the OO-implementation.

### Model
The model code consists of a _layer_naive_dense_ and a _naive_model_sequential_ function.

#### The OO-Implementation

```R
layer_naive_dense <- function(input_size, output_size, activation) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"

  random_array <- function(dim, min = 0, max = 1){
  array(runif(prod(dim), min, max),dim)
  }

  self$activation <- activation

  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)

  b_shape <- c(output_size)
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)

  self$weights <- list(self$W, self$b)

  self$call <- function(inputs) {
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }
  self
}

naive_model_sequential <- function(layers) {
  self <- new.env(parent = emptyenv())
  self$layers <- layers

  weights <- lapply(layers, function(layer) layer$weights)
  self$weights <- do.call(c, weights)

  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }
  self
}
```

The user can afterwards initialize a model and generate predictions with the following code:

```R
model <- naive_model_sequential(
  list(
    layer_naive_dense(input_size = 28 * 28, output_size = 512, activation = tf$nn$relu),
    layer_naive_dense(input_size = 512, output_size = 10, activation = tf$nn$softmax)
))

predictions <- model$call(images_batch)
```

This code enables the user to chain as many _naive_dense_layers_ after each other as he/she wants. The weights are stored in each layer and the layer structure is stored in the _naive_model_sequential_. If I want to avoid using local environments that means I need to manage the layer structure as well as the layer weights completely outside of this code. Lets go!

#### Functional implementation

```R
layer_naive_dense <- function(inputs, W, b, activation) {
  activation(tf$matmul(inputs, W) + b)
}

naive_model_sequential <- function(inputs, layer_params){
  x <- inputs
  for (layer in layer_params){
    x <- layer_naive_dense(inputs = x, W = layer$W, b =  layer$b, activation = layer$activation)
  }
  return(x)
}

initialize_layers <- function(layer_config, min = 0, max = 0.1){
  layers_params <- list()

  random_array <- function(dim, min = 0, max = 1){
    array(runif(prod(dim), min, max),dim)
  }

  for(i in 1:length(layer_config)){
    w_shape <- c(layer_config[[i]]$input_size, layer_config[[i]]$output_size)
    w_initial_value <- random_array(w_shape, min = min, max = max)

    b_shape <- c(layer_config[[i]]$output_size)
    b_initial_value <- array(0, b_shape)

    layers_params[[i]] <- list(
      W = tf$Variable(w_initial_value),
      b = tf$Variable(b_initial_value),
      activation = layer_config[[i]]$activation)
  }
  return(layers_params)
}
```

The code for the _layer_naive_dense_ and the _naive_model_sequential_ is much shorter here than in the OO implementation. However this is not enough. In order to initialize the weights for an arbitrary amount of layers we need to add the _initialize_layers_ function. Having written the functions above, the user can now create the model with:

```R
layer_params <- initialize_layers(
  list(
    list(input_size = 28*28, output_size = 512, activation = tf$nn$relu),
    list(input_size = 512, output_size = 10, activation = tf$nn$softmax)
  )
)

predictions <- naive_model_sequential(inputs = images_batch, layer_params = layer_params)
```

#### Differences

| Aspect | Conclusion | Winner | 
|----------|----------|----------|
| Clarity of code | Object-oriented style is slightly more verbose because of the flattening of weights that happens in the _naive_model_sequential_ function. Object properties need to be defined inside the function. Otherwise it is pretty similar | Tie |
| Ease of use | Basically the same. The only difference is that in the functional implementation the user needs to pass the layer_params to the model. | Tie |
| Flexibility | The functional implementation is more flexible because we can easily control how we want to initialize the weights. That is not possible in the object-oriented implementation. | Functional |
| Debugability | In the OO implementation, we can quickly extract meta data regarding the model generator from the model object: layers and weights. In the functional implementation this info is available in the current environment. | Tie |

#### Conclusion

Again, the functional implementation is the winner. So functional over OO? Wait! Too early! We forgot the most important part: How easy is it to train each model?

### Model training

The model training part is where the magic happens. Here, we calculate gradients and update the model weights based on gradient descent. This implementation uses the automatic differentiation capabilities of tensorflow`s _GradientTape_ to do that. Furthermore, we do everything in mini-batches and making sure that our training loop loops over the whole training dataset for one epoch.

#### Object oriented implementation

```R
one_training_step <- function(model, images_batch, labels_batch) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model$call(images_batch)
    per_sample_losses <-
      loss_sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss <- mean(per_sample_losses)
  })
  gradients <- tape$gradient(average_loss, model$weights)
  update_weights(gradients, model$weights)
  average_loss
}

update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub(gradients[[i]] * 1e-3)
}

fit_model <- function(model, images, labels, epochs, batch_size = 128, test_images, test_labels) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels)
    for (batch_counter in seq_len(batch_generator$num_batches)) {
      batch <- batch_generator$get_next_batch()
      loss <- one_training_step(model, batch$images, batch$labels)
    }
    predictions <- model$call(test_images)
    predicted_labels <- max.col(predictions) - 1
    matches <- predicted_labels == test_labels
    cat(sprintf("accuracy: %.2f\n", mean(matches)))
  }
}
```

Afterwards, the user can do the training and generate predictions like this:

```R
layer_params_learned <- fit_model(model, train_images, train_labels, epochs = 1, batch_size = 128)

predictions <- model(test_images)
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == as.array(test_labels)
cat(sprintf("accuracy: %.2f\n", mean(matches)))
```

#### Functional implementation

```R
one_training_step <- function(model, images_batch, labels_batch, layer_params) {
  with(tf$GradientTape() %as% tape, {
    predictions <- naive_model_sequential(inputs = images_batch, layer_params = layer_params)
    per_sample_loss <- loss_sparse_categorical_crossentropy(y_true = labels_batch, y_pred = predictions)
    average_loss <- mean(per_sample_loss)
  })
  weights <- lapply(layer_params, function(layer_params) list(layer_params$W, layer_params$b))
  weights <- do.call(c, weights)
  gradients <- tape$gradient(average_loss, weights)
  weights <- update_weights(gradients, weights)
  j <- 1
  for(i in seq(from = 1, to = length(layer_params)*2 -1, by = 2)){
    layer_params[[(i +1) / 2]]$W <- weights[[i]]
    layer_params[[(i +1) / 2]]$b <- weights[[i + 1]]
  }
  return(layer_params)
}

update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub(gradients[[i]] * 1e-3)
  return(weights)
}

fit_model <- function(model, layer_params, images, labels, epochs, batch_size = 128, test_images, test_labels) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    num_batches <- ceiling(nrow(images) / batch_size)
    for(batch_counter in seq_len(num_batches)){
      batch <- new_batch_generator(images, labels, batch_counter)
      layer_params <- one_training_step(model, batch$images, batch$labels, layer_params)
    }
    predictions <- naive_model_sequential(test_images, layer_params)
    predicted_labels <- max.col(predictions) - 1
    matches <- predicted_labels == as.array(test_labels)
    cat(sprintf("accuracy: %.2f\n", mean(matches)))
  }
  return(layer_params)
}
```

Afterwards, the user can do the training and generate predictions like this:

```R
layer_params_learned <- fit_model(layer_params = layer_params_initial, train_images, train_labels, epochs = 1, batch_size = 128)

predictions <- naive_model_sequential(test_images, layer_params_learned)
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == as.array(test_labels)
cat(sprintf("accuracy: %.2f\n", mean(matches)))
```

The _fit_model_ and the _update_weights_ functions are basically the same. The difference is in the _one_training_step_ function. The functional implementation is longer and more complicated than the object-oriented one. Managing the weights outside of the model creates a problem: The model requires the weights to be in a nested structure (representing layers) while the gradient calculation expects a flattened weight structure. To solve the issue, we need to first flatten the nested weights, then update them and then nest them again.
We cannot use the model to pass us the weights in the good format since the model itself is stateless and we want to manage the weights outside of the model.
If we define the weights as a flat object in the beginning, we will have the hassle of creating nested weights in the _naive_model_sequential_ function so we don`t win something.

#### Differences

| Aspect | Conclusion | Winner | 
|----------|----------|----------|
| Clarity of code | Functional implementation is more complicated and longer. | OO |
| Ease of use | The only difference is that in the functional implementation the needs to make an extra step to save the weights after the _fit_model_ function and pass them to the model to generate predictions. | OO |
| Flexibility | Same flexibility. | Tie |
| Debugability | In the OO implementation, we can quickly extract meta data regarding the model generator from the model object: layers and weights. In the functional implementation this info is available in the current environment. | Tie |

#### Conclusion

Now the clear winner is the object-oriented implementation.
Writing the model in the object-oriented way where weights are stored in the model object itself allows us to add calculations inside the model to transform the weights for an outside receiver.

So altough our model code is more complicated, our model training code becomes more easy. A general phenomenon seems to emerge: Object-oriented implementations take away complexity for higher level code (like in a machine learning training loop) and hide that complexity in lower-level code because they can perform additional compuations that are being managed in the local environments.

## The complete functional implementation

Having written down each part of the functional implementation above, here is again the full functional low-level implementation using naive dense layers and a naive sequential model:

```R
library(keras)
library(tensorflow)

layer_naive_dense <- function(inputs, W, b, activation) {
  activation(tf$matmul(inputs, W) + b)
}

initialize_layers <- function(layer_config, min = 0, max = 0.1){
  layers_params <- list()

  random_array <- function(dim, min = 0, max = 1){
    array(runif(prod(dim), min, max),dim)
  }

  for(i in 1:length(layer_config)){
    w_shape <- c(layer_config[[i]]$input_size, layer_config[[i]]$output_size)
    w_initial_value <- random_array(w_shape, min = min, max = max)

    b_shape <- c(layer_config[[i]]$output_size)
    b_initial_value <- array(0, b_shape)

    layers_params[[i]] <- list(
      W = tf$Variable(w_initial_value),
      b = tf$Variable(b_initial_value),
      activation = layer_config[[i]]$activation)
  }
  return(layers_params)
}

naive_model_sequential <- function(inputs, layer_params){
  x <- inputs
  for (layer in layer_params){
    x <- layer_naive_dense(inputs = x, W = layer$W, b =  layer$b, activation = layer$activation)
  }
  return(x)
}

new_batch_generator <- function(images, labels, batch_size = 128, batch_id = 1) {
  start <- batch_size * (batch_id -1) + 1
  end <- batch_size * batch_id
  if(end >= nrow(images)) {
    end <- nrow(images)
  }
  indices <- start:end
  return(list(images = images[indices, ], labels = labels[indices]))
}

one_training_step <- function(model, images_batch, labels_batch, layer_params) {
  with(tf$GradientTape() %as% tape, {
    predictions <- naive_model_sequential(inputs = images_batch, layer_params = layer_params)
    per_sample_loss <- loss_sparse_categorical_crossentropy(y_true = labels_batch, y_pred = predictions)
    average_loss <- mean(per_sample_loss)
  })
  weights <- lapply(layer_params, function(layer_params) list(layer_params$W, layer_params$b))
  weights <- do.call(c, weights)

  gradients <- tape$gradient(average_loss, weights)
  weights <- update_weights(gradients, weights)
  j <- 1
  for(i in seq(from = 1, to = length(layer_params)*2 -1, by = 2)){
    layer_params[[(i +1) / 2]]$W <- weights[[i]]
    layer_params[[(i +1) / 2]]$b <- weights[[i + 1]]
  }
  return(layer_params)
}

update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub(gradients[[i]] * 1e-3)
  return(weights)
}

fit_model <- function(model, layer_params, images, labels, epochs, batch_size = 128, test_images, test_labels) {
  for (epoch_counter in seq_len(epochs)) {
    cat("Epoch ", epoch_counter, "\n")
    num_batches <- ceiling(nrow(images) / batch_size)
    for(batch_counter in seq_len(num_batches)){
      batch <- new_batch_generator(images, labels, batch_counter)
      layer_params <- one_training_step(model, batch$images, batch$labels, layer_params)
    }
    predictions <- naive_model_sequential(test_images, layer_params)
    predicted_labels <- max.col(predictions) - 1
    matches <- predicted_labels == as.array(test_labels)
    cat(sprintf("accuracy: %.2f\n", mean(matches)))
  }
  return(layer_params)
}
```

After we load the functions above we can train a model to classify MNIST with the following code:

```R
mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28)) / 255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28)) / 255
test_labels <- mnist$test$y
train_labels <- mnist$train$y

layer_params_initial <- initialize_layers(
  layer_config = list(
    list(input_size = 28*28, output_size = 512, activation = tf$nn$relu),
    list(input_size = 512, output_size = 10, activation = tf$nn$softmax)
  )
)

layer_params_learned <- fit_model(train_images, train_labels, epochs = 1, batch_size = 128, layer_params = layer_params_initial)
```

In order to evaluate the model we need to run:

```R
predictions <- model(test_images, layer_params_learned)
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == as.array(test_labels)
cat(sprintf("accuracy: %.2f\n", mean(matches)))
```

## Final conclusion

It seems that writing object-oriented code allows us to push complexity to lower-level functions.

If we were to implement an entire machine-learning framework ourselves, it might not matter whether we use an object-oriented or a functional approach since we have to do the calculations somewhere - either in the basic or in the high-level functions.

But fortunately we dont have to write everything! We mostly need to write high level code and we can rely on very robust and standard low level implementations. Since those low-level implementations have their local environments, they can take away some of the complexity for us in a way that a purely functional implementation cannot. The example above was a very basic one. If we would add more complexity to our model, for example by adding dropout or regularization, the object-oriented implementation would shine even more since high level code would still be easy while it would get much more complicated in the functional implementation.

Taking into account that a user typically only needs to write high level functions, the overall winner for me is the object-oriented implementation. I finally learned, why it makes sense to implement machine-learning frameworks in the object-oriented way!