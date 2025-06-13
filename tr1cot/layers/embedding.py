import functools
import math

import tensorflow as tf

import mlable.layers.embedding

# CONSTANTS ####################################################################

# meta
EPSILON = 1e-5
DROPOUT = 0.0

# CONTEXT ######################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class RateEmbeddingBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        shift_dim: int=1,
        wave_dim: int=10000,
        **kwargs
    ) -> None:
        # init
        super(RateEmbeddingBlock, self).__init__(**kwargs)
        # save for IO serialization
        self._config = {
            'embed_dim': embed_dim,
            'shift_dim': shift_dim,
            'wave_dim': wave_dim,}
        # layers
        self._embed = None
        self._proj0 = None
        self._proj1 = None

    def build(self, inputs_shape: tuple) -> None:
        # init
        self._embed = mlable.layers.embedding.CosineEmbedding(**self.get_cosine_config())
        self._proj0 = tf.keras.layers.Dense(activation='silu', **self.get_dense_config())
        self._proj1 = tf.keras.layers.Dense(activation=None, **self.get_dense_config())
        # build
        __shape = tuple(inputs_shape)
        self._embed.build(__shape)
        __shape = self._embed.compute_output_shape(__shape)
        self._proj0.build(__shape)
        __shape = self._proj0.compute_output_shape(__shape)
        self._proj1.build(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # embed (B, ...) => (B, ..., E)
        __outputs = self._embed(inputs)
        # transform (B, .., E) => (B, ..., E)
        return self._proj1(self._proj0(__outputs))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape) + (self._config['embed_dim'],)

    def get_config(self) -> dict:
        __config = super(RateEmbeddingBlock, self).get_config()
        __config.update(self._config)
        return __config

    def get_cosine_config(self) -> dict:
        return dict(self._config)

    def get_dense_config(self) -> dict:
        return {'units': self._config['embed_dim'], 'use_bias': True, 'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)