import tensorflow as tf
import keras.backend as K
from keras import activations,initializers,regularizers,constraints
from keras.layers import Input, Dense, Layer, InputSpec


class TransposableDense(tf.keras.layers.Dense):

  def __init__(self, units, **kwargs):
    super().__init__(units, **kwargs)

  def build(self, input_shape):
    assert len(input_shape) >= 2
    input_dim = input_shape[-1]
    self.t_output_dim = input_dim

    self.kernel = self.add_weight(shape=(int(input_dim), self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(shape=(self.units,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
      self.bias_t = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='bias_t',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
    else:
      self.bias = None
      self.bias_t = None
    # self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
    self.built = True

  def call(self, inputs, transpose=False):
    bs, input_dim = inputs.get_shape()

    kernel = self.kernel
    bias = self.bias
    if transpose:
      assert input_dim == self.units
      kernel = tf.keras.backend.transpose(kernel)
      bias = self.bias_t

    output = tf.keras.backend.dot(inputs, kernel)
    if self.use_bias:
      output = tf.keras.backend.bias_add(output, bias, data_format='channels_last')
    if self.activation is not None:
      output = self.activation(output)
    return output

  def compute_output_shape(self, input_shape):
    bs, input_dim = input_shape
    output_dim = self.units
    if input_dim == self.units:
      output_dim = self.t_output_dim
    return bs, output_dim