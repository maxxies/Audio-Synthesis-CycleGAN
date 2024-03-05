import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D,LayerNormalization, Multiply, Activation, LSTM



def multi_head_self_attention(inputs, num_heads=8):

    batch_size, seq_length, depth = tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape().as_list()[-1]
    depth_per_head = depth // num_heads
    
    def split_heads(x):
        return tf.transpose(tf.reshape(x, (batch_size, seq_length, num_heads, depth_per_head)), perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        return tf.matmul(attention_weights, v)
    
    qkv = tf.keras.layers.Dense(depth * 3)(inputs)
    q, k, v = tf.split(qkv, 3, axis=-1)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    
    attention_output = scaled_dot_product_attention(q, k, v)
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, (batch_size, seq_length, depth))
    
    # Optionally, apply additional dense layer on top
    output = tf.keras.layers.Dense(depth)(attention_output)
    
    return output + inputs  # Skip connection


def gated_linear_layer(inputs, gates, name=None):
    activation = Multiply()([inputs, Activation('sigmoid')(gates)])
    if name is not None:
        activation._name = name
    return activation


def instance_norm_layer(inputs, epsilon=1e-06, activation_fn=None, name=None):
    instance_norm_layer = LayerNormalization(epsilon=epsilon, name=name)(inputs)
    if activation_fn is not None:
        instance_norm_layer = Activation(activation_fn)(instance_norm_layer)
    return instance_norm_layer

def conv1d_layer(inputs, filters, kernel_size, strides=1, padding='same', activation=None, kernel_initializer=None, name=None):
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=kernel_initializer, name=name)(inputs)
    return conv_layer

def conv2d_layer(inputs, filters, kernel_size, strides, padding='same', activation=None, kernel_initializer=None, name=None):
    conv_layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=kernel_initializer, name=name)(inputs)
    return conv_layer

def residual1d_block(inputs, filters=1024, kernel_size=3, strides=1, name_prefix='residual_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=name_prefix + 'h2_norm')

    h3 = tf.keras.layers.add([inputs, h2_norm], name=name_prefix + 'add')

    return h3

def residual1d_block_with_attention(inputs, filters=1024, kernel_size=3, strides=1, name_prefix='residule_block_'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    
    # Integrate self-attention
    h1_attention = multi_head_self_attention(h1_glu)
    
    h2 = conv1d_layer(inputs=h1_attention, filters=filters // 2, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3


def downsample1d_block(inputs, filters, kernel_size, strides, name_prefix='downsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def downsample2d_block(inputs, filters, kernel_size, strides, name_prefix='downsample2d_block_'):
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample1d_block(inputs, filters, kernel_size, strides, shuffle_size=2, name_prefix='upsample1d_block_'):
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(inputs, shape=[n, ow, oc], name=name)

    return outputs

def generator_gatedcnn(inputs, reuse=False, scope_name='generator_gatedcnn'):
    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.compat.v1.variable_scope(scope_name, reuse=reuse) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=256, kernel_size=5, strides=2, name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block2_')
        r3 = residual1d_block(inputs=r2, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block3_')
        r4 = residual1d_block(inputs=r3, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block4_')
        r5 = residual1d_block(inputs=r4, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block5_')

        # Upsample
        u1 = upsample1d_block(inputs=r5, filters=1024, kernel_size=5, strides=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=512, kernel_size=5, strides=2, name_prefix='upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2

def generator_gatedcnn_attention(inputs, reuse=False, scope_name='generator_gatedcnn-attention'):
    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.compat.v1.variable_scope(scope_name, reuse=reuse) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=256, kernel_size=5, strides=2, name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block_with_attention(inputs=d2, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block_with_attention1_')
        r2 = residual1d_block_with_attention(inputs=r1, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block_with_attention2_')
        r3 = residual1d_block_with_attention(inputs=r2, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block_with_attention3_')
        r4 = residual1d_block_with_attention(inputs=r3, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block_with_attention4_')
        r5 = residual1d_block_with_attention(inputs=r4, filters=1024, kernel_size=3, strides=1, name_prefix='residual1d_block_with_attention5_')

        # Upsample
        u1 = upsample1d_block(inputs=r5, filters=1024, kernel_size=5, strides=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=512, kernel_size=5, strides=2, name_prefix='upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2

def discriminator(inputs, reuse=False, scope_name='discriminator'):
      # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.compat.v1.variable_scope(scope_name, reuse=reuse) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], activation=None, name='h1_conv')
        h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[3, 3], strides=[2, 2], name_prefix='downsample2d_block1_')
        d2 = downsample2d_block(inputs=d1, filters=512, kernel_size=[3, 3], strides=[2, 2], name_prefix='downsample2d_block2_')
        d3 = downsample2d_block(inputs=d2, filters=1024, kernel_size=[6, 3], strides=[1, 2], name_prefix='downsample2d_block3_')

        # Output
        o1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(d3)

        return o1
