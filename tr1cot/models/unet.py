import functools
import itertools

import tensorflow as tf

import mlable.blocks.convolution.unet
import mlable.layers.embedding
import mlable.models.diffusion
import mlable.utils

import tr1cot.layers.embedding

# CONSTANTS ####################################################################

DROPOUT_RATE = 0.0
EPSILON_RATE = 1e-6

START_RATE = 0.98 # signal rate at the start of the forward diffusion process
END_RATE = 0.02 # signal rate at the end of the forward diffusion process

# UTILS ########################################################################

def cycle(data: any) -> iter:
    __data = data if mlable.utils.iterable(data) else [data]
    return itertools.cycle(__data)

# DIFFUSION ####################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class UnetDiffusionModel(mlable.models.diffusion.LatentDiffusionModel):
    def __init__(
        self,
        channel_dim: iter,
        group_dim: iter,
        head_dim: iter,
        head_num: iter=4,
        layer_num: iter=1,
        add_attention: iter=False,
        add_downsampling: iter=False,
        add_upsampling: iter=False,
        start_rate: float=START_RATE,
        end_rate: float=END_RATE,
        dropout_rate: float=DROPOUT_RATE,
        epsilon_rate: float=EPSILON_RATE,
        **kwargs
    ) -> None:
        # init
        super(UnetDiffusionModel, self).__init__(start_rate=start_rate, end_rate=end_rate, **kwargs)
        # the length of the channels list will be the reference for all the args
        __block_dim = list(channel_dim) if mlable.utils.iterable(channel_dim) else [channel_dim]
        # save for IO serialization
        self._config.update({
            'channel_dim': __block_dim,
            'group_dim': cycle(group_dim),
            'head_dim': cycle(head_dim),
            'head_num': cycle(head_num),
            'layer_num': cycle(layer_num),
            'add_attention': cycle(add_attention),
            'add_downsampling': cycle(add_downsampling),
            'add_upsampling': cycle(add_upsampling),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),})
        # layers
        self._embed_rate = None
        self._embed_height = None
        self._embed_width = None
        self._unet_blocks = []

    def build(self, inputs_shape: tuple) -> None:
        __shape_o, __shape_c = tuple(tuple(__i) for __i in inputs_shape)
        # save the shape of the data
        super(UnetDiffusionModel, self).build(__shape_o)
        # embed (B,) => (B, E)
        self._embed_rate = tr1cot.layers.embedding.RateEmbeddingBlock(
            name='diffusion-0-embed-rate',
            **self.get_embed_rate_config())
        # embed height (B, H, W, L) => (B, H, W, L)
        self._embed_height = mlable.layers.embedding.PositionalEmbedding(
            name='diffusion-1-embed-height',
            **self.get_embed_height_config())
        # embed width (B, H, W, L) => (B, H, W, L)
        self._embed_width = mlable.layers.embedding.PositionalEmbedding(
            name='diffusion-2-embed-width',
            **self.get_embed_width_config())
        # Transform (B, Hi, Wi, Li) => (B, Hi+1, Wi+1, Li+1)
        self._unet_blocks = [
            mlable.blocks.convolution.unet.UnetBlock(
                name=f'diffusion-{__i + 3}-unet',
                **__c)
            for __i, __c in enumerate(self.get_unet_configs())]
        # build
        self._embed_rate.build(__shape_c)
        __shape_c = self._embed_rate.compute_output_shape(__shape_c)
        self._embed_height.build(__shape_o)
        __shape_o = self._embed_height.compute_output_shape(__shape_o)
        self._embed_width.build(__shape_o)
        __shape_o = self._embed_width.compute_output_shape(__shape_o)
        for __b in self._unet_blocks:
            __b.build(inputs_shape=__shape_o, contexts_shape=__shape_c)
            __shape_o = __b.compute_output_shape(inputs_shape=__shape_o, contexts_shape=__shape_c)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        __cast = functools.partial(tf.cast, dtype=self.compute_dtype)
        # separate the data from the context in the inputs
        __outputs, __contexts = tuple(__cast(__i) for __i in inputs)
        # embed the noise rate (B,) => (B, E)
        __contexts = self._embed_rate(__contexts)
        # embed the spatial dimensions (B, H, W, L) => (B, H, W, L)
        __outputs = self._embed_width(self._embed_height(__outputs))
        # transform the data (B, Hi, Wi, Li) => (B, Hi+1, Wi+1, Li+1)
        return functools.reduce(lambda __x, __b: __b(__x, contexts=__contexts, training=training, **kwargs), self._unet_blocks, __outputs)

    def compute_output_shape(self, inputs_shape: tuple) -> tuple:
        return tuple(inputs_shape[0])

    def get_config(self) -> dict:
        __config = super(UnetDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    def get_embed_rate_config(self) -> dict:
        return {'embed_dim': self._config['channel_dim'][0], 'wave_dim': 10000, 'shift_dim': 1,}

    def get_embed_height_config(self) -> dict:
        return {'sequence_axis': 1, 'feature_axis': -1,}

    def get_embed_width_config(self) -> dict:
        return {'sequence_axis': 2, 'feature_axis': -1,}

    def get_unet_configs(self) -> dict:
        return [{
            'channel_dim': __c,
            'group_dim': __g,
            'head_dim': __h,
            'head_num': __n,
            'layer_num': __l,
            'add_attention': __a,
            'add_downsampling': __d,
            'add_upsampling': __u,
            'dropout_rate': self._config['dropout_rate'],
            'epsilon_rate': self._config['epsilon_rate'],}
        for __c, __g, __h, __n, __l, __a, __d, __u in zip(
            self._config['channel_dim'],
            self._config['group_dim'],
            self._config['head_dim'],
            self._config['head_num'],
            self._config['layer_num'],
            self._config['add_attention'],
            self._config['add_downsampling'],
            self._config['add_upsampling'])]

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
