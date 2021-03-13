import os
import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.training import moving_averages
from PIL import Image, ImageDraw

################### Mask ###################
def random_bbox(img_height=256, img_width=256, margins=0, mask_size=128, random_mask=True):
    """Generate a random tlhw with configuration.

    Args:
        img_height: height of image.
        img_width: width of image.
        margins: margins of mask and image border.
        mask_size: size of mask.
        random_mask: if True, random location. if False, central location.

    Returns:
        tuple: (top, left, height, width)

    """
    if random_mask is True:
        maxt = img_height - margins - mask_size
        maxl = img_width - margins - mask_size
        t = tf.random_uniform(
            [], minval=margins, maxval=maxt, dtype=tf.int32)
        l = tf.random_uniform(
            [], minval=margins, maxval=maxl, dtype=tf.int32)
    else:
        t = (img_height - mask_size)//2
        l = (img_width - mask_size)//2
    h = tf.constant(mask_size)
    w = tf.constant(mask_size)
    return (t, l, h, w)

def bbox2mask(bbox, img_height=256, img_width=256, max_delta=32, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        img_height: height of image.
        img_width: width of image.
        max_delta: max delta of masks.
        name: name of variable scope.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta//2+1)
        w = np.random.randint(delta//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        mask = tf.py_func(
            npmask,
            [bbox, img_height, img_width, max_delta],
            tf.float32, stateful=False)
        mask.set_shape([1] + [img_height, img_width] + [1])
    return mask

def brush_stroke_mask(img_height=256, img_width=256, name='mask'):
    """Generate free form mask tensor.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W, 1))
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        mask = tf.py_func(
            generate_mask,
            [img_height, img_width],
            tf.float32, stateful=True)
        mask.set_shape([1] + [img_height, img_width] + [1])
    return mask

################### Visualization ###################
def vqvae_visual(gt, recons, iter, size, save_dir):
    """ Show 4 reconstruction images in the training of VQVAE.""" 

    # gap between each images
    gap = 2

    # height and width of result image
    height = size * 4 + gap * 3
    width = size * 2 + gap * 1

    result = 255 * np.ones((height, width, 3), dtype=np.uint8) 

    for i in range(4):
        gt_i = ((gt[i] + 1.) * 127.5).astype(np.uint8)
        recons_i = ((recons[i] + 1.) * 127.5).astype(np.uint8)
        
        # fill the images into grid
        result[i*(size+gap):i*(size+gap)+size, 0*(size+gap):0*(size+gap)+size, ::-1] = gt_i
        result[i*(size+gap):i*(size+gap)+size, 1*(size+gap):1*(size+gap)+size, ::-1] = recons_i
        
    cv2.imwrite(os.path.join(save_dir, 'vqvae%d.png' % iter), result)

def structure_visual(gt, masked, recons_gen, recons_gt, iter, size, save_dir):
    """ Show 4 generated structure feature maps in the training of structure generator.""" 

    # gap between each images
    gap = 2

    # height and width of result image
    height = size * 4 + gap * 3
    width = size * 4 + gap * 3

    result = 255 * np.ones((height, width, 3), dtype=np.uint8) 

    for i in range(4):
        gt_i = ((gt[i] + 1.) * 127.5).astype(np.uint8)
        masked_i = ((masked[i] + 1.) * 127.5).astype(np.uint8)
        recons_gen_i = ((recons_gen[i] + 1.) * 127.5).astype(np.uint8)
        recons_gt_i = ((recons_gt[i] + 1.) * 127.5).astype(np.uint8)
        
        # fill the images into grid
        result[i*(size+gap):i*(size+gap)+size, 0*(size+gap):0*(size+gap)+size, ::-1] = gt_i
        result[i*(size+gap):i*(size+gap)+size, 1*(size+gap):1*(size+gap)+size, ::-1] = masked_i
        result[i*(size+gap):i*(size+gap)+size, 2*(size+gap):2*(size+gap)+size, ::-1] = recons_gen_i
        result[i*(size+gap):i*(size+gap)+size, 3*(size+gap):3*(size+gap)+size, ::-1] = recons_gt_i
        
    cv2.imwrite(os.path.join(save_dir, 'structure%d.png' % iter), result)

def texture_visual(gt, masked, complete, recons_gt, iter, size, save_dir):
    """ Show 4 inpainting images under guidance of ground truth structure feature maps in the training of texture generator."""

    # gap between each images
    gap = 2

    # height and width of result image
    height = size * 4 + gap * 3
    width = size * 4 + gap * 3

    result = 255 * np.ones((height, width, 3), dtype=np.uint8) 

    for i in range(4):
        gt_i = ((gt[i] + 1.) * 127.5).astype(np.uint8)
        masked_i = ((masked[i] + 1.) * 127.5).astype(np.uint8)
        complete_i = ((complete[i] + 1.) * 127.5).astype(np.uint8)
        recons_gt_i = ((recons_gt[i] + 1.) * 127.5).astype(np.uint8)
        
        # fill the images into grid
        result[i*(size+gap):i*(size+gap)+size, 0*(size+gap):0*(size+gap)+size, ::-1] = masked_i
        result[i*(size+gap):i*(size+gap)+size, 1*(size+gap):1*(size+gap)+size, ::-1] = recons_gt_i
        result[i*(size+gap):i*(size+gap)+size, 2*(size+gap):2*(size+gap)+size, ::-1] = complete_i
        result[i*(size+gap):i*(size+gap)+size, 3*(size+gap):3*(size+gap)+size, ::-1] = gt_i
        
    cv2.imwrite(os.path.join(save_dir, 'texture%d.png' % iter), result)

################### Utilities ###################
def feature_loss(f, idx, embed, softmax_scale=10.):
    fs = f.get_shape().as_list()
    embedding_dim = fs[-1]
    num_embeddings = embed.get_shape().as_list()[-1]
    flat_f = tf.reshape(f, [-1, embedding_dim])
    d = (tf.reduce_sum(flat_f**2, 1, keepdims=True)
        - 2 * tf.matmul(flat_f, embed)
        + tf.reduce_sum(embed ** 2, 0, keepdims=True))
    d_mean, d_var = tf.nn.moments(d, 1, keep_dims=True)
    d_std = d_var**0.5
    d_score = -1 * tf.nn.tanh((d - d_mean) / d_std) 
    d_score = tf.reshape(d_score, fs[:-1] + [-1])
    encoding = tf.one_hot(idx, num_embeddings) 
    ce = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_scale*d_score, labels=encoding)
    loss = tf.reduce_mean(ce)
    return loss
    
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
    
def int_shape(x):
    return list(map(int, x.get_shape()))

def hw_flatten(x) :
    return tf.reshape(x, [x.shape[0], -1, x.shape[-1]])

def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def get_var_maybe_avg(var_name, ema, **kwargs):
    """ utility for retrieving polyak averaged params """
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_name(layer_name, counters):
    """ utlity for keeping track of layer names """
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

################### Base layers ###################
@add_arg_scope
def conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, counters={}, ema=None, **kwargs):
    """ convolutional layer """
    name = get_name('conv2d', counters)
    xs = int_shape(x)
    # See https://arxiv.org/abs/1502.03167v3.
    input_feature_size = filter_size[0]*filter_size[1]*xs[3]
    stddev = 1. / math.sqrt(input_feature_size)
    with tf.variable_scope(name):
        W = get_var_maybe_avg('W', ema, shape=filter_size+[xs[3],num_filters], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(0, stddev), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, counters={}, ema=None, **kwargs):
    """ transposed convolutional layer """
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    # See https://arxiv.org/abs/1502.03167v3.
    input_feature_size = filter_size[0]*filter_size[1]*xs[3]
    stddev = 1. / math.sqrt(input_feature_size)
    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    with tf.variable_scope(name):
        W = get_var_maybe_avg('W', ema, shape=filter_size+[num_filters,xs[3]], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(0, stddev), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x, W, target_shape, [1] + stride + [1], padding=pad)
        x = tf.nn.bias_add(x, b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def wndense(x, num_units, nonlinearity=None, counters={}, ema=None, **kwargs):
    """ dense layer (weight norm) """
    name = get_name('wndense', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=[int(x.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def wnconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], rate=1, pad='SAME', nonlinearity=None, counters={}, ema=None, **kwargs):
    """ convolutional layer (weight norm) """
    name = get_name('wnconv2d', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad, dilations=[1,rate,rate,1]), b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = wndense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])

@add_arg_scope
def gated_conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], rate=1, pad='SAME', nonlinearity=tf.nn.elu, counters={}, **kwargs):
    """ gated convolutional layer """
    name = get_name('gated_conv2d', counters)
    xs = int_shape(x)
    # See https://arxiv.org/abs/1502.03167v3.
    input_feature_size = filter_size[0]*filter_size[1]*xs[3]
    stddev = 1. / math.sqrt(input_feature_size)
    with tf.variable_scope(name):
        W = tf.get_variable('W', shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(0, stddev), trainable=True)
        b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.), trainable=True)

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad, dilations=[1,rate,rate,1]), b)
        
        x, y = tf.split(x, 2, 3)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        y = tf.nn.sigmoid(y)
        x = x * y

        return x

@add_arg_scope
def gated_deconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=tf.nn.elu, counters={}, **kwargs):
    """ upsample and gated convolutional layer """
    name = get_name('gated_deconv2d', counters)
    x = tf.image.resize_nearest_neighbor(x, [2*int(x.get_shape()[1]), 2*int(x.get_shape()[2])], align_corners=True)
    xs = int_shape(x)
    # See https://arxiv.org/abs/1502.03167v3.
    input_feature_size = filter_size[0]*filter_size[1]*xs[3]
    stddev = 1. / math.sqrt(input_feature_size)
    with tf.variable_scope(name):
        W = tf.get_variable('W', shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(0, stddev), trainable=True)
        b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.), trainable=True)

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)
        
        x, y = tf.split(x, 2, 3)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)
            
        y = tf.nn.sigmoid(y)
        x = x * y

        return x

@add_arg_scope
def snconv2d(x, num_filters, filter_size=[5,5], stride=[2,2], pad='SAME', nonlinearity=None, counters={}, **kwargs):
    """ convolutional layer (spectral norm) """
    name = get_name('snconv2d', counters)
    xs = int_shape(x)
    with tf.variable_scope(name):
        W = tf.get_variable('W', shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(0, 0.05), trainable=True)
        u = tf.get_variable('u', shape=[1, num_filters], dtype=tf.float32, 
                            initializer=tf.truncated_normal_initializer(), trainable=False)
        b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.), trainable=True)

        # spectral normalization
        w_mat = tf.reshape(W, [-1, num_filters])
        v_ = tf.matmul(u, tf.transpose(w_mat))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w_mat)
        u_hat = l2_norm(u_)
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_mat, W.get_shape().as_list())

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, w_norm, [1] + stride + [1], pad), b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

################### Meta-layer consisting of multiple base layers ###################
@add_arg_scope
def vector_quantize(inputs, embedding_dim=64, num_embeddings=512, commitment_cost=0.25, decay=0.99, epsilon=1e-5, is_training=False, counters={}, **kwargs):
    """ vector quantizer """
    name = get_name('vector_quantize', counters)
    with tf.variable_scope(name):
        # w is a matrix with an embedding in each column. When training, the embedding
        # is assigned to be the average of all inputs assigned to that embedding.
        w = tf.get_variable('embedding', [embedding_dim, num_embeddings],
                            initializer=tf.random_normal_initializer(), use_resource=True)
        ema_cluster_size = tf.get_variable('ema_cluster_size', [num_embeddings],
                            initializer=tf.constant_initializer(0), use_resource=True)
        ema_w = tf.get_variable('ema_dw', initializer=w.initialized_value(), use_resource=True)

        with tf.control_dependencies([inputs]):
            w_value = w.read_value()
        input_shape = tf.shape(inputs)
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], 
                                        embedding_dim), [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, embedding_dim])

        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                    - 2 * tf.matmul(flat_inputs, w_value)
                    + tf.reduce_sum(w_value ** 2, 0, keepdims=True))

        encoding_indices = tf.argmax(- distances, 1)
        encodings = tf.one_hot(encoding_indices, num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        with tf.control_dependencies([encoding_indices]):
            w_lookup = tf.transpose(w_value, [1, 0])
        quantized = tf.nn.embedding_lookup(w_lookup, encoding_indices, validate_indices=False)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

        if is_training:
            updated_ema_cluster_size = moving_averages.assign_moving_average(ema_cluster_size, 
                                                            tf.reduce_sum(encodings, 0), decay)
            dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
            updated_ema_w = moving_averages.assign_moving_average(ema_w, dw, decay)
            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + epsilon)
                                / (n + num_embeddings * epsilon) * n)

            normalised_updated_ema_w = (updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
            with tf.control_dependencies([e_latent_loss]):
                update_w = tf.assign(w, normalised_updated_ema_w)
                with tf.control_dependencies([update_w]):
                    loss = commitment_cost * e_latent_loss

        else:
            loss = commitment_cost * e_latent_loss

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        
        return quantized, loss, encoding_indices, w.read_value()

@add_arg_scope
def resnet(x, num_res_channel=64, nonlinearity=tf.nn.elu, conv=conv2d, **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_res_channel)
    c2 = conv(nonlinearity(c1), num_filters, filter_size=[1,1])

    return x + c2

@add_arg_scope
def cond_resnet(x, nonlinearity=concat_elu, conv=wnconv2d, rate=1, **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters * 2, rate=rate)

    a, b = tf.split(c1, 2, 3)
    c2 = a * tf.nn.sigmoid(b)

    return x + c2

@add_arg_scope
def out_resnet(x, nonlinearity=tf.nn.elu, conv=nin, **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters)

    return x + c1

@add_arg_scope
def gated_resnet(x, a=None, h=None, num_res_filters=128, nonlinearity=concat_elu, conv=wnconv2d, rate=1, dropout_p=0., causal_attention=False, num_head=8, **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_res_filters, rate=rate)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_res_filters)
    c1 = nonlinearity(c1)
    
    c2 = conv(c1, num_filters * 2, rate=1)
    if h is not None: # add condition h if included: conditional generation
        c2 += nin(nonlinearity(h), num_filters * 2)

    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)

    if causal_attention:
        canvas_size = int(np.prod(int_shape(c3)[1:-1]))
        causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
        for i in range(canvas_size):
            causal_mask[i, :i] = 1.
        causal_mask = tf.constant(causal_mask, dtype=tf.float32)
        causal_mask = tf.expand_dims(causal_mask, axis=0)

        multihead_src = []
        for head_rep in range(num_head):
            query = nin(c3, num_filters//8)
            key = nin(c3, num_filters//8)
            value = nin(c3, num_filters//2)

            dot = tf.matmul(hw_flatten(query), hw_flatten(key), transpose_b=True)/np.sqrt(num_filters//8)
            dot = dot - (1. - causal_mask) * 1e10 # masked softmax
            dot = dot - tf.reduce_max(dot, axis=-1, keep_dims=True)
            causal_exp_dot = tf.exp(dot) * causal_mask
            causal_probs = causal_exp_dot / (tf.reduce_sum(causal_exp_dot, axis=-1, keep_dims=True) + 1e-10)
            atten = tf.matmul(causal_probs, hw_flatten(value))
            atten = tf.reshape(atten, [xs[0], xs[1], xs[2], -1])
            multihead_src.append(atten)

        multihead = tf.concat(multihead_src, axis=-1)
        multihead = nin(multihead, num_filters)
        c3 = c3 + multihead
    
    if dropout_p > 0:
        c3 = tf.nn.dropout(c3, keep_prob=1. - dropout_p)

    return x + c3
  
# "Generative Image Inpainting with Contextual Attention" https://github.com/JiahuiYu/generative_inpainting
# "PEPSI: Fast Image Inpainting With Parallel Decoding Network" https://github.com/Forty-lock/PEPSI
def attention_transfer(f, b1, b2, ksize=3, stride=1, fuse_k=3, softmax_scale=50., fuse=False):
    # extract patches from background feature maps with rate (1st scale)
    bs1 = tf.shape(b1)
    int_bs1 = b1.get_shape().as_list()
    w_b1 = tf.extract_image_patches(b1, [1,4,4,1], [1,4,4,1], [1,1,1,1], padding='SAME')
    w_b1 = tf.reshape(w_b1, [int_bs1[0], -1, 4, 4, int_bs1[3]])
    w_b1 = tf.transpose(w_b1, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # extract patches from background feature maps with rate (2nd scale)
    bs2 = tf.shape(b2)
    int_bs2 = b2.get_shape().as_list()
    w_b2 = tf.extract_image_patches(b2, [1,2,2,1], [1,2,2,1], [1,1,1,1], padding='SAME')
    w_b2 = tf.reshape(w_b2, [int_bs2[0], -1, 2, 2, int_bs2[3]])
    w_b2 = tf.transpose(w_b2, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # use structure feature maps as foreground for matching and use background feature maps for reconstruction.
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    w_f = tf.extract_image_patches(f, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w_f = tf.reshape(w_f, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w_f = tf.transpose(w_f, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    
    w_f_groups = tf.split(w_f, int_fs[0], axis=0)
    w_b1_groups = tf.split(w_b1, int_bs1[0], axis=0)
    w_b2_groups = tf.split(w_b2, int_bs2[0], axis=0)
    y1 = []
    y2 = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw1_wi, raw2_wi in zip(f_groups, w_f_groups, w_b1_groups, w_b2_groups):
        # conv for compare
        wi = wi[0] #(k,k,c,hw)
        onesi = tf.ones_like(wi)
        xxi = tf.nn.conv2d(tf.square(xi), onesi, strides=[1,1,1,1], padding="SAME") #(1,h,w,hw)
        wwi = tf.reduce_sum(tf.square(wi), axis=[0,1,2], keep_dims=True) #(1,1,1,hw)
        xwi = tf.nn.conv2d(xi, wi, strides=[1,1,1,1], padding="SAME") #(1,h,w,hw)
        di = xxi + wwi - 2*xwi
        di_mean, di_var = tf.nn.moments(di, 3, keep_dims=True)
        di_std = di_var**0.5
        yi = -1 * tf.nn.tanh((di - di_mean) / di_std)

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], fs[1]*fs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], fs[1], fs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], fs[1]*fs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], fs[2], fs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], fs[1]*fs[2]])

        # softmax to match
        yi = tf.nn.softmax(yi*scale, 3)

        wi_center1 = raw1_wi[0]
        wi_center2 = raw2_wi[0]
        y1.append(tf.nn.conv2d_transpose(yi, wi_center1, tf.concat([[1], bs1[1:]], axis=0), strides=[1,4,4,1]))
        y2.append(tf.nn.conv2d_transpose(yi, wi_center2, tf.concat([[1], bs2[1:]], axis=0), strides=[1,2,2,1]))

    y1 = tf.concat(y1, axis=0)
    y2 = tf.concat(y2, axis=0)
   
    return y1, y2

################### Shifting the image around, efficient alternative to masking convolutions ###################
def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return wnconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return wnconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
 
