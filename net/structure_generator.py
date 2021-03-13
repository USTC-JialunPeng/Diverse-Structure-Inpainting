import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import net.nn as nn

def structure_condition_spec(x, mask, ema=None, nr_channel=32, nr_res_channel=32, resnet_nonlinearity='concat_elu'):
    """
    Input:
    Tensor x of shape (N,H,W,3) (e.g. (4,256,256,3))
    Tensor mask of shape (N,H,W,1) (e.g. (4,256,256,1))
    Output:
    Tensor cond of shape (N,H//8,W//8,C') (e.g. (4,32,32,256))
    """

    counters = {}
    with arg_scope([nn.wnconv2d], counters=counters, ema=ema):

        # Parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity):
            ones_x = tf.ones_like(x)[:, :, :, 0:1]
            x = tf.concat([x, ones_x, ones_x*mask], 3)

            x = nn.wnconv2d(x, nr_channel, filter_size=[5,5])
            x = nn.gated_resnet(x, num_res_filters=nr_res_channel, conv=nn.wnconv2d)
            x = nn.wnconv2d(x, 2*nr_channel, filter_size=[3,3], stride=[2,2])
            x = nn.gated_resnet(x, num_res_filters=2*nr_res_channel, conv=nn.wnconv2d)
            x = nn.wnconv2d(x, 4*nr_channel, filter_size=[3,3], stride=[2,2])
            x = nn.gated_resnet(x, num_res_filters=4*nr_res_channel, conv=nn.wnconv2d)
            x = nn.wnconv2d(x, 8*nr_channel, filter_size=[3,3], stride=[2,2])
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d, rate=2)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d, rate=4)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d, rate=8)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d, rate=16)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d)
            x = nn.gated_resnet(x, num_res_filters=8*nr_res_channel, conv=nn.wnconv2d)
            cond = x

            return cond

def structure_pixelcnn_spec(e, h=None, ema=None, dropout_p=0., nr_resnet=20, nr_out_resnet=20, nr_channel=128, nr_res_channel=128, nr_attention=4, nr_head=8, resnet_nonlinearity='concat_elu', num_embeddings=512):
    """
    Input:
    Tensor e of shape (N,H//8,W//8,C) (e.g. (4,32,32,64))
    Tensor h of shape (N,H//8,W//8,C') (e.g. (4,32,32,256))
    Output:
    Tensor e_out of shape (N,H//8,W//8,K) (e.g. (4,32,32,512))
    """

    counters = {}
    with arg_scope([nn.wnconv2d, nn.wndense], counters=counters, ema=ema):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], num_res_filters=nr_res_channel, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, num_head=nr_head, h=h):
            # PixelCNN
            es = nn.int_shape(e)
            e_pad = tf.concat([e, tf.ones(es[:-1]+[1])], 3) # add channel of ones to distinguish image from padding later on
            u = nn.down_shift(nn.down_shifted_conv2d(e_pad, num_filters=nr_channel, filter_size=[2, 3])) # stream for pixels above
            ul = nn.down_shift(nn.down_shifted_conv2d(e_pad, num_filters=nr_channel, filter_size=[1, 3])) + \
                 nn.right_shift(nn.down_right_shifted_conv2d(e_pad, num_filters=nr_channel, filter_size=[2, 1])) # stream for up and to the left

            for attn_rep in range(nr_attention):
                for rep in range(nr_resnet // nr_attention - 1):
                    u = nn.gated_resnet(u, conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, u, conv=nn.down_right_shifted_conv2d)

                u = nn.gated_resnet(u, conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, u, conv=nn.down_right_shifted_conv2d, causal_attention=True)

            for out_rep in range(nr_out_resnet):
                ul = nn.out_resnet(ul, conv=nn.nin)

            e_out = nn.nin(tf.nn.elu(ul), num_embeddings)

            return e_out
