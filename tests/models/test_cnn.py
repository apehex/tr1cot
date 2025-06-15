import functools

import tensorflow as tf

import mlable.shapes
import tokun.models.vqvae
import tr1cot.models.cnn

# UNET #########################################################################

class CnnDiffusionModelTest(tf.test.TestCase):
    def setUp(self):
        super(CnnDiffusionModelTest, self).setUp()
        # test cases
        self._cases = [
            {
                'tokun': '../tokun/models/vqvae.4x64.keras',
                'inputs': tf.random.uniform((2, 8, 8, 4), minval=0, maxval=256, dtype=tf.int32),
                'latents': tf.random.normal((2, 8, 8, 256), dtype=tf.float32),
                'variances': tf.random.uniform((2, 1, 1, 1), minval=0.0, maxval=1.0, dtype=tf.float32),
                'args': {'block_num': 2, 'latent_dim': [128], 'start_rate': 0.95, 'end_rate': 0.05,},
                'shapes': {'matched': (2, 8, 8, 1), 'extended': (2, 8, 8, 256), 'concatenated': (2, 8, 8, 512), 'embedded': (2, 8, 8, 512), 'evened': (2, 8, 8, 128), 'compressed': (2, 8, 8, 128), 'transformed': (2, 8, 8, 128), 'decompressed': (2, 8, 8, 128), 'projected': (2, 8, 8, 256),},},
            {
                'tokun': '../tokun/models/vqvae.1x64.keras',
                'inputs': tf.random.uniform((2, 8, 8, 1), minval=0, maxval=256, dtype=tf.int32),
                'latents': tf.random.normal((2, 8, 8, 64), dtype=tf.float32),
                'variances': tf.random.uniform((2, 1, 1, 1), minval=0.0, maxval=1.0, dtype=tf.float32),
                'args': {'block_num': 2, 'latent_dim': [64, 128], 'start_rate': 0.95, 'end_rate': 0.05,},
                'shapes': {'matched': (2, 8, 8, 1), 'extended': (2, 8, 8, 64), 'concatenated': (2, 8, 8, 128), 'embedded': (2, 8, 8, 128), 'evened': (2, 8, 8, 64), 'compressed': (2, 4, 4, 128), 'transformed': (2, 4, 4, 128), 'decompressed': (2, 8, 8, 64), 'projected': (2, 8, 8, 64),},},]

    def test_internals(self):
        for __case in self._cases:
            # parse
            __batch_dim, __height_dim, __width_dim, __feature_dim = tuple(__case['latents'].shape)
            # init
            __model = tr1cot.models.cnn.CnnDiffusionModel(**__case['args'])
            # build
            __model((__case['latents'], __case['variances']), training=False)
            # extend
            self.assertEqual((1, __feature_dim), tuple(__model._extend_block.weights[0].shape))
            self.assertEqual((__feature_dim,), tuple(__model._extend_block.weights[-1].shape))
            # embed
            self.assertEqual((__height_dim, 2 * __feature_dim), __model._embed_height_block._layer.embeddings.shape)
            self.assertEqual((__width_dim, 2 * __feature_dim), __model._embed_width_block._layer.embeddings.shape)
            # even
            self.assertEqual((2 * __feature_dim, __case['args']['latent_dim'][0]), tuple(__model._even_block.weights[0].shape))
            self.assertEqual((__case['args']['latent_dim'][0],), tuple(__model._even_block.weights[-1].shape))
            # encode
            self.assertEqual(len(__case['args']['latent_dim'][1:]), len(__model._encode_blocks))
            # transform
            self.assertEqual(__case['args']['block_num'], len(__model._transform_blocks))
            # decode
            self.assertEqual(len(__case['args']['latent_dim'][1:]), len(__model._decode_blocks))
            # project
            self.assertEqual((__case['args']['latent_dim'][0], __feature_dim), tuple(__model._project_block.weights[0].shape))
            self.assertEqual((__feature_dim,), tuple(__model._project_block.weights[-1].shape))

    def test_shapes_and_dtypes(self):
        for __case in self._cases:
            # parse
            __batch_dim, __height_dim, __width_dim, __feature_dim = tuple(__case['latents'].shape)
            # init
            __model = tr1cot.models.cnn.CnnDiffusionModel(**__case['args'])
            # end-to-end
            __outputs = __model((__case['latents'], __case['variances']), training=False)
            self.assertEqual(tuple(__case['shapes']['projected']), tuple(__outputs.shape))
            self.assertEqual(tuple(__case['latents'].shape), tuple(__outputs.shape))
            self.assertEqual(tf.float32, __outputs.dtype)
            # unpack
            __outputs, __variances = __case['latents'], __case['variances']
            # match (B, 1, 1, 1) => (B, H, W, 1)
            __variances = __model._match_block(__variances)
            self.assertEqual(__case['shapes']['matched'], tuple(__variances.shape))
            # extend (B, H, W, 1) => (B, H, W, E)
            __variances = __model._extend_block(__variances)
            self.assertEqual(__case['shapes']['extended'], tuple(__variances.shape))
            # merge (B, H, W, E) + (B, H, W, E) => (B, H, W, 2*E)
            __outputs = __model._concat_block([__outputs, __variances])
            self.assertEqual(__case['shapes']['concatenated'], tuple(__outputs.shape))
            # embed the spatial axes (B, H, W, E+1)
            __outputs = __model._embed_height_block(__outputs)
            __outputs = __model._embed_width_block(__outputs)
            self.assertEqual(__case['shapes']['embedded'], tuple(__outputs.shape))
            # expand (B, H, W, E+1) => (B, H, W, L)
            __outputs = __model._even_block(__outputs)
            self.assertEqual(__case['shapes']['evened'], tuple(__outputs.shape))
            # save residuals that skip the whole sampling process
            __residuals = __outputs
            # downsample (B, Hi, Wi, Li) => (B, Hi/2, Wi/2, Li+1)
            __outputs = functools.reduce(lambda __x, __b: __b(__x, training=False), __model._encode_blocks, __outputs)
            self.assertEqual(__case['shapes']['compressed'], tuple(__outputs.shape))
            # transform (B, Hn, Wn, Ln) => (B, Hn, Wn, Ln)
            __outputs = functools.reduce(lambda __x, __b: __b(__x, training=False), __model._transform_blocks, __outputs)
            self.assertEqual(__case['shapes']['transformed'], tuple(__outputs.shape))
            # upsample (B, Hi, Wi, Li) => (B, 2Hi, 2Wi, Li-1)
            __outputs = functools.reduce(lambda __x, __b: __b(__x, training=False), __model._decode_blocks, __outputs)
            self.assertEqual(__case['shapes']['decompressed'], tuple(__outputs.shape))
            # project (B, H, W, L) => (B, H, W, E)
            __outputs = __model._project_block(__outputs + __residuals)
            self.assertEqual(__case['shapes']['projected'], tuple(__outputs.shape))

    def test_latent_space_is_normal(self):
        for __case in self._cases:
            # parse
            __batch_dim, __height_dim, __width_dim, __feature_dim = tuple(__case['latents'].shape)
            # init
            __model = tr1cot.models.cnn.CnnDiffusionModel(**__case['args'])
            __tokun = tf.keras.models.load_model(__case['tokun'], compile=False)
            # build
            __model.set_vae(__tokun, trainable=False)
            __model((__case['latents'], __case['variances']), training=False)
            __tokun(__case['inputs'], training=False)
            # encode into the latent space
            __latents = __model.preprocess(__case['inputs'])
            # compute mean and std deviation
            __mean = tf.math.reduce_mean(__latents)
            __sigma = tf.math.reduce_std(__latents)
            # compare to the normal distribution
            self.assertEqual(0, int(tf.round(10.0 * __mean)))
            self.assertEqual(100, int(tf.round(100.0 * __sigma)))

    def test_sample_generation(self):
        for __case in self._cases:
            # parse
            __batch_dim, __height_dim, __width_dim, __feature_dim = tuple(__case['latents'].shape)
            # init
            __model = tr1cot.models.cnn.CnnDiffusionModel(**__case['args'])
            __tokun = tf.keras.models.load_model(__case['tokun'], compile=False)
            # build
            __model.set_vae(__tokun, trainable=False)
            __model((__case['latents'], __case['variances']), training=False)
            __tokun(__case['inputs'], training=False)
            # generate
            __samples = __model.generate(sample_num=2, step_num=4, logits=True)
            self.assertEqual((2, __height_dim, __width_dim, 8 * tuple(__case['inputs'].shape)[-1]), tuple(__samples.shape))
