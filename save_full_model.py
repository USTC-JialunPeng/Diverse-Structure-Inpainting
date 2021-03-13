import os
import sys
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
import skimage.measure

from net.vqvae import vq_encoder_spec, vq_decoder_spec
from net.structure_generator import structure_condition_spec, structure_pixelcnn_spec
from net.texture_generator import texture_generator_spec, texture_discriminator_spec
import net.nn as nn

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Data
parser.add_argument('--checkpoint_dir', default='model_logs/', type=str,
                    help='directory of tensorflow checkpoint.')
parser.add_argument('--structure_generator_dir', type=str, default='/gdata/vqvae-inpainting/20200821-030240_celebahq_StructureGenerator', 
                    help='pre-trained structure generator is given here.')
parser.add_argument('--texture_generator_dir', type=str, default='/gdata/vqvae-inpainting/20200819-025515_celebahq_TextureGenerator', 
                    help='pre-trained texture generator is given here.')

# Architecture
parser.add_argument('--image_size', type=int, default=256,
                    help='provide square images of this size.')
parser.add_argument('--nr_channel_vq', type=int, default=128,
                    help='number of channels in VQVAE.')
parser.add_argument('--nr_res_block_vq', type=int, default=2,
                    help='number of residual blocks in VQVAE.') 
parser.add_argument('--nr_res_channel_vq', type=int, default=64,
                    help='number of channels in the residual block in VQVAE.')
parser.add_argument('--nr_channel_s', type=int, default=128, 
                    help='number of channels in structure pixelcnn.')
parser.add_argument('--nr_res_channel_s', type=int, default=128,
                    help='number of channels in the residual block in structure pixelcnn.')
parser.add_argument('--nr_resnet_s', type=int, default=20, 
                    help='number of residual blocks in structure pixelcnn.')
parser.add_argument('--nr_resnet_out_s', type=int, default=20, 
                    help='number of output residual blocks in structure pixelcnn.')
parser.add_argument('--nr_attention_s', type=int, default=4, 
                    help='number of attention blocks in structure pixelcnn.')
parser.add_argument('--nr_head_s', type=int, default=8, 
                    help='number of attention heads in attention blocks.')
parser.add_argument('--nr_channel_cond_s', type=int, default=32, 
                    help='number of channels in structure condition network.')
parser.add_argument('--nr_res_channel_cond_s', type=int, default=32, 
                    help='number of channels in the residual block of structure condition network.')
parser.add_argument('--resnet_nonlinearity', type=str, default='concat_elu', 
                    help='nonlinearity in structure generator. One of "concat_elu", "elu", "relu". ')
parser.add_argument('--nr_channel_gen_t', type=int, default=64, 
                    help='number of channels in texture generator.')
parser.add_argument('--nr_channel_dis_t', type=int, default=64, 
                    help='number of channels in texture discriminator.')
  
# Vector quantizer
parser.add_argument('--embedding_dim', type=int, default=64, 
                    help='number of the dimensions of embeddings in vector quantizer.')
parser.add_argument('--num_embeddings', type=int, default=512, 
                    help='number of embeddings in vector quantizer.')
parser.add_argument('--commitment_cost', type=float, default=0.25,
                    help='weight of commitment loss in vector quantizer.')
parser.add_argument('--decay', type=float, default=0.99,
                    help='decay of EMA updates in vector quantizer.')

# EMA setting
parser.add_argument('--ema_decay', type=float, default=0.9997, 
                    help='decay rate of EMA in validation.')

args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

################### Build structure generator & texture generator ###################
# Create VQVAE network
vq_encoder = tf.make_template('vq_encoder', vq_encoder_spec)
vq_encoder_opt = {'nr_channel': args.nr_channel_vq, 
                  'nr_res_block': args.nr_res_block_vq,
                  'nr_res_channel': args.nr_res_channel_vq,
                  'embedding_dim': args.embedding_dim,
                  'num_embeddings': args.num_embeddings,
                  'commitment_cost': args.commitment_cost,
                  'decay': args.decay}

vq_decoder = tf.make_template('vq_decoder', vq_decoder_spec)
vq_decoder_opt = {'nr_channel': args.nr_channel_vq, 
                  'nr_res_block': args.nr_res_block_vq,
                  'nr_res_channel': args.nr_res_channel_vq,
                  'embedding_dim': args.embedding_dim}

# Create structure generator
structure_condition = tf.make_template('structure_condition', structure_condition_spec)
structure_condition_opt = {'nr_channel': args.nr_channel_cond_s, 
                           'nr_res_channel': args.nr_res_channel_cond_s, 
                           'resnet_nonlinearity': args.resnet_nonlinearity}

structure_pixelcnn = tf.make_template('structure_pixelcnn', structure_pixelcnn_spec)
structure_pixelcnn_opt = {'nr_channel': args.nr_channel_s,
                          'nr_res_channel': args.nr_res_channel_s,
                          'nr_resnet': args.nr_resnet_s,
                          'nr_out_resnet': args.nr_resnet_out_s,
                          'nr_attention': args.nr_attention_s,
                          'nr_head': args.nr_head_s,
                          'resnet_nonlinearity': args.resnet_nonlinearity,
                          'num_embeddings': args.num_embeddings}

# Create texture generator
texture_generator = tf.make_template('texture_generator', texture_generator_spec)
texture_generator_opt = {'nr_channel': args.nr_channel_gen_t}

texture_discriminator = tf.make_template('texture_discriminator', texture_discriminator_spec)
texture_discriminator_opt = {'nr_channel': args.nr_channel_dis_t}

# Full model
img_ph = tf.placeholder(tf.float32, shape=(1, args.image_size, args.image_size, 3))
mask_ph = tf.placeholder(tf.float32, shape=(1, args.image_size, args.image_size, 1))
e_sample = tf.placeholder(tf.float32, shape=(1, args.image_size//8, args.image_size//8, args.embedding_dim))
h_sample = tf.placeholder(tf.float32, shape=(1, args.image_size//8, args.image_size//8, 8*args.nr_channel_cond_s))

batch_pos = img_ph
mask = mask_ph
masked = batch_pos * (1. - mask)
enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
dec_gt = vq_decoder(enc_gt['quant_t'], enc_gt['quant_b'], **vq_decoder_opt)
cond_masked = structure_condition(masked, mask, **structure_condition_opt)
pix_out = structure_pixelcnn(e_sample, h_sample, dropout_p=0., **structure_pixelcnn_opt)
gen_out = texture_generator(masked, mask, e_sample, **texture_generator_opt)
dis_out = texture_discriminator(tf.concat([batch_pos, mask], axis=3), **texture_discriminator_opt)

# Variables to restore
ema = tf.train.ExponentialMovingAverage(decay=args.ema_decay)
structure_generator_params = []
for v in tf.trainable_variables():
    if 'structure' in v.name:
        structure_generator_params.append(v)
variables_to_restore = ema.variables_to_restore(moving_avg_variables=structure_generator_params)
structure_variables_to_restore = {}
else_variables_to_restore = {}
for item in variables_to_restore:
    if 'structure' in item:
        structure_variables_to_restore[item] = variables_to_restore[item]
    else:
        else_variables_to_restore[item] = variables_to_restore[item]

################### Evaluate test images ###################
# Create a saver to restore structure generator
restore_structure_saver = tf.train.Saver(structure_variables_to_restore)

# Create a saver to restore VQVAE & texture generator
restore_else_saver = tf.train.Saver(else_variables_to_restore)

# Create a saver to save full model
saver = tf.train.Saver()

# TF session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Restore structure generator
    ckpt = tf.train.get_checkpoint_state(args.structure_generator_dir)
    if ckpt and ckpt.model_checkpoint_path:
        restore_structure_saver.restore(sess, ckpt.model_checkpoint_path)
        print('Structure generator restored ...')
    else:
        print('Restore structure generator failed! EXIT!')
        sys.exit()

    # Restore VQVAE & texture generator
    ckpt = tf.train.get_checkpoint_state(args.texture_generator_dir)
    if ckpt and ckpt.model_checkpoint_path:
        restore_else_saver.restore(sess, ckpt.model_checkpoint_path)
        print('VQVAE & texture generator restored ...')
    else:
        print('Restore VQVAE & texture generator failed! EXIT!')
        sys.exit()

    # Save full model
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path)
    print('Full model saved.')
    sys.stdout.flush()
