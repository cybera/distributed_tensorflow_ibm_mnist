from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_parameter_mgr
import numpy as np

FLAGS = tf.app.flags.FLAGS
batch_size=tf_parameter_mgr.getTrainBatchSize()

IMAGE_SIZE = 28
#IMAGE_PIXELS = 784  ???? why its converting to RGB ????
IMAGE_PIXELS = 2352
IMAGE_LAYERS = 3 ##???? why

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = tf_parameter_mgr.getLearningRateDecay()  # Learning rate decay factor.
INITIAL_LEARNING_RATE = tf_parameter_mgr.getBaseLearningRate()       # Initial learning rate.

def read_mnist(filename_queue):

  reader = tf.TFRecordReader()
  _, value = reader.read(filename_queue)
  features = tf.parse_single_example(value,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
  label = tf.cast(features['label'], tf.int32)

  #result.label = tf.expand_dims(result.label, 0)
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([IMAGE_PIXELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  return image, label


def inputs(eval_data, batch_size):
 
  if not eval_data:
    filenames = tf_parameter_mgr.getTrainData()
    #num_epochs = NUM_EPOCHS_TRAIN
  else:
    filenames = tf_parameter_mgr.getTestData()
    #num_epochs = NUM_EPOCHS_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  image, label = read_mnist(filename_queue)
  print("image Shape")
  print(image.get_shape())
  # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
  images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
           # Ensures a minimum amount of shuffling of examples.
       min_after_dequeue=1000)

  return images, sparse_labels

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images):
  """Build the mnist model.

  Args:
    images: Images returned from  inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  
  x_image = tf.reshape(images, [-1, 28, 28, IMAGE_LAYERS])

  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, IMAGE_LAYERS, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')

    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                   name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 32, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    n_dim = np.prod(pool2.get_shape().as_list()[1:])
    #reshape = tf.reshape(pool2, [-1, 7 * 7 * 64])
    reshape = tf.reshape(pool2, [-1, n_dim])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #_activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
     weights = _variable_with_weight_decay('weights', shape=[1024, 192],
                                              stddev=0.04, wd=0.004)
     biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
     local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
                            name=scope.name)
    #_activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, 10],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [10],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    #_activation_summary(softmax_linear)

  return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  print("Total loss")
  print(total_loss)

  return total_loss

def accuracy(logits, labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  correct = np.sum(top_k_op)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  print("Accuracy")
  print(accuracy)
  return accuracy


def train(total_loss, global_step, return_grad = False):
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

  loss_averages_op = _add_loss_summaries(total_loss)

  with tf.control_dependencies([loss_averages_op]):
    opt = tf_parameter_mgr.getOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  print("train")
  if return_grad: return apply_gradient_op, grads
  return train_op


def _add_loss_summaries(total_loss):
  """Add summaries for losses in mnist model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  return loss_averages_op



