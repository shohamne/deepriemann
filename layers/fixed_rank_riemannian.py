from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
#from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.summary import summary

from tensorflow.python.layers import base
#from tensorflow.python.layers import utils


from pymanopt import manifolds

class FixedRankRiemannian(base._Layer):  # pylint: disable=protected-access
  """Fixed-Rank-Densely-connected layer class.
  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a fixed rank weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).
  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.
  Arguments:
    units: Integer or Long, dimensionality of the output space.
    rank: Integer or Long, rank of the weights matrix
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the weight matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, units, rank,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               summaries = [],
               **kwargs):
    super(FixedRankRiemannian, self).__init__(trainable=trainable, name=name, **kwargs)
    self.units = units
    self.rank = rank
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.summaries = summaries

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape.ndims is None:
      raise ValueError('Inputs to `Dense` should have known rank.')
    if len(input_shape) < 2:
      raise ValueError('Inputs to `Dense` should have rank >= 2.')
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # Note that we set `trainable=True` because this is a trainable
    # weight of the layer. If the layer is not trainable
    # (self.trainable = False), the variable will not be added to
    # tf.trainable_variables(), and self.trainable_weights will be empty.
    m = input_shape[-1].value
    k = self.rank
    n = self.units


    # glorot unifiorm
    limit = (6.0 / (m + n))**0.5
    kernel0 = random_ops.random_uniform([m,n],minval=-limit,maxval=limit)
    s0, u0, v0 = linalg_ops.svd(kernel0,full_matrices=False)
    v0 = array_ops.transpose(v0)
    u0 = array_ops.slice(u0,[0,0],[m,k]) #u0[:,:k]
    s0 = array_ops.slice(s0,[0,],[k,]) #s0[:k]
    v0 = array_ops.slice(v0,[0,0],[k,n]) #v0[:k,:]

    self.manifold_args = [
                   vs.get_variable('U',
                                 #shape=[m, k],
                                 initializer=u0,#init_ops.orthogonal_initializer(),
                                 regularizer=self.kernel_regularizer,
                                 dtype=self.dtype,
                                 trainable=True),
                   vs.get_variable('S',
                                   #shape=[k,],
                                   initializer=s0,#self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   dtype=self.dtype,
                                   trainable=True),
                   vs.get_variable('V',
                                    #shape=[k, n],
                                    initializer=v0,#init_ops.orthogonal_initializer(),
                                    regularizer=self.kernel_regularizer,
                                    dtype=self.dtype,
                                    trainable=True)]

    U = self.manifold_args[0]
    US = standard_ops.matmul(U, standard_ops.diag(self.manifold_args[1]))
    USV = standard_ops.matmul(US, self.manifold_args[2])
    self.kernel = USV

    if 'norm' in self.summaries:
        summary.scalar('krenel-norm', linalg_ops.norm(self.kernel))
    if 'histogram' in self.summaries:
        summary.histogram('kernel-histogram', self.kernel)

    manifold = manifolds.FixedRankEmbedded(m,n,k)

    if self.use_bias:
      self.bias = vs.get_variable('bias',
                                shape=[n],
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                dtype=self.dtype,
                                trainable=True)

      self.manifold_args.append(self.bias)
      manifold = manifolds.Product([manifold, manifolds.Euclidean(n)])

    else:
      self.bias = None

    self.manifold = manifold

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    output_shape = shape[:-1] + [self.units]
    if len(output_shape) > 2:
      raise ValueError( 'len(output_shape) > 2 is not supported' )
    else:
      xU = standard_ops.matmul(inputs, self.manifold_args[0])
      xUS = standard_ops.matmul(xU, standard_ops.diag(self.manifold_args[1]))
      xUSV = standard_ops.matmul(xUS, self.manifold_args[2])
      outputs = xUSV

    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

def fixed_rank_riemannian(
    inputs, units, rank,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    summaries=[],
    name=None,
    reuse=None):
  """Functional interface for the fixed-rank-densely-connected layer.
  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).
  Note: if the `inputs` tensor has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.
  Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    summaries: List of internal quantities to visualize on tensorboard, should be a subset of['norm', 'histogram']
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Returns:
    Output tensor.
  """
  layer = FixedRankRiemannian(units, rank,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                summaries = summaries,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs), layer.manifold, layer.manifold_args
