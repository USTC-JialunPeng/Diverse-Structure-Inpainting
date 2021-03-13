import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import net.nn as nn

def vq_encoder_spec(x, ema=None, nr_channel=128, nr_res_block=2, nr_res_channel=64, embedding_dim=64, 
                    num_embeddings=512, commitment_cost=0.25, decay=0.99, is_training=False):
    """
    Input:
    Tensor x of shape (N,H,W,3) (e.g. (128,256,256,3))
    Output:
    Tensor enc_t of shape (N,H//8,W//8,C) (e.g. (128,32,32,64))
    Tensor enc_b of shape (N,H//4,W//4,C) (e.g. (128,64,64,64))
    Tensor quant_t of shape (N,H//8,W//8,C) (e.g. (128,32,32,64))
    Tensor quant_b of shape (N,H//4,W//4,C) (e.g. (128,64,64,64))
    Tensor loss of shape (1,) 
    Tensor idx_t of shape (N,H//8,W//8) (e.g. (128,32,32))
    Tensor idx_b of shape (N,H//4,W//4) (e.g. (128,64,64))
    Tensor embed_t of shape (C,K) (e.g. (64,512))
    Tensor embed_b of shape (C,K) (e.g. (64,512))
    """
    
    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.vector_quantize], counters=counters, ema=ema):

        # Bottom encoder
        enc_b = nn.conv2d(x, nr_channel//2, filter_size=[4,4], stride=[2,2])
        enc_b = tf.nn.elu(enc_b)
        enc_b = nn.conv2d(enc_b, nr_channel, filter_size=[4,4], stride=[2,2])
        enc_b = tf.nn.elu(enc_b)
        enc_b = nn.conv2d(enc_b, nr_channel)
        for rep in range(nr_res_block):
            enc_b = nn.resnet(enc_b, num_res_channel=nr_res_channel, nonlinearity=tf.nn.elu)
        enc_b = tf.nn.elu(enc_b)

        # Top encoder
        enc_t = nn.conv2d(enc_b, nr_channel//2, filter_size=[4,4], stride=[2,2])
        enc_t = tf.nn.elu(enc_t)
        enc_t = nn.conv2d(enc_t, nr_channel)
        for rep in range(nr_res_block):
            enc_t = nn.resnet(enc_t, num_res_channel=nr_res_channel, nonlinearity=tf.nn.elu)
        enc_t = tf.nn.elu(enc_t)
        enc_t = nn.conv2d(enc_t, embedding_dim, filter_size=[1,1])

        # Vector quantization with top codebook
        quant_t, diff_t, idx_t, embed_t = nn.vector_quantize(enc_t, embedding_dim=embedding_dim, 
                                    num_embeddings=num_embeddings, commitment_cost=commitment_cost, 
                                    decay=decay, is_training=is_training)

        # Top decoder
        dec_t = nn.conv2d(quant_t, nr_channel)
        for rep in range(nr_res_block):
            dec_t = nn.resnet(dec_t, num_res_channel=nr_res_channel, nonlinearity=tf.nn.elu)
        dec_t = tf.nn.elu(dec_t)
        dec_t = nn.deconv2d(dec_t, nr_channel, filter_size=[4,4], stride=[2,2])
        enc_b = tf.concat([enc_b, dec_t], -1)
        enc_b = nn.conv2d(enc_b, embedding_dim, filter_size=[1,1])

        # Vector quantization with bottom codebook
        quant_b, diff_b, idx_b, embed_b = nn.vector_quantize(enc_b, embedding_dim=embedding_dim, 
                                    num_embeddings=num_embeddings, commitment_cost=commitment_cost, 
                                    decay=decay, is_training=is_training)

        return {'enc_t': enc_t, 'enc_b': enc_b, 'quant_t': quant_t, 'quant_b': quant_b, 'loss': diff_t + diff_b,
                'idx_t': idx_t, 'idx_b': idx_b, 'embed_t': embed_t, 'embed_b': embed_b}

def vq_decoder_spec(quant_t, quant_b, ema=None, nr_channel=128, nr_res_block=2, nr_res_channel=64, embedding_dim=64):
    """
    Input:
    Tensor quant_t of shape (N,H//8,W//8,C) (e.g. (128,32,32,64))
    Tensor quant_b of shape (N,H//4,W//4,C) (e.g. (128,64,64,64))
    Output:
    Tensor dec_b of shape (N,H,W,3) (e.g. (128,256,256,3))
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d], counters=counters, ema=ema):

        # Bottom decoder
        quant_t = nn.deconv2d(quant_t, embedding_dim, filter_size=[4,4], stride=[2,2])
        dec_b = tf.concat([quant_b, quant_t], -1)
        dec_b = nn.conv2d(dec_b, nr_channel)
        for rep in range(nr_res_block):
            dec_b = nn.resnet(dec_b, num_res_channel=nr_res_channel, nonlinearity=tf.nn.elu)
        dec_b = tf.nn.elu(dec_b)
        dec_b = nn.deconv2d(dec_b, nr_channel//2, filter_size=[4,4], stride=[2,2])
        dec_b = tf.nn.elu(dec_b)
        dec_b = nn.deconv2d(dec_b, 3, filter_size=[4,4], stride=[2,2])

        return {'dec_b': dec_b}
