import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from data.data_loader import DataLoader
from net.vqvae import vq_encoder_spec, vq_decoder_spec
from net.structure_generator import structure_condition_spec, structure_pixelcnn_spec
import net.nn as nn

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Data
parser.add_argument('--checkpoints_dir', type=str, default='/gdata/vqvae-inpainting', 
                    help='checkpoints are saved here.')
parser.add_argument('--dataset', type=str, default='celebahq', 
                    help='dataset of the experiment.')
parser.add_argument('--train_flist', type=str, default='/gdata/celeba-hq/train.flist', 
                    help='file list of training set.')
parser.add_argument('--valid_flist', type=str, default='/gdata/celeba-hq/val.flist', 
                    help='file list of validation set.')
parser.add_argument('--vqvae_network_dir', type=str, default='/gdata/vqvae-inpainting/20200805-190115_celebahq_VQVAE', 
                    help='pre-trained VQVAE network is given here.')

# Architecture
parser.add_argument('--load_size', type=int, default=266,
                    help='scale images to this size.')
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
  
# Vector quantizer
parser.add_argument('--embedding_dim', type=int, default=64, 
                    help='number of the dimensions of embeddings in vector quantizer.')
parser.add_argument('--num_embeddings', type=int, default=512, 
                    help='number of embeddings in vector quantizer.')
parser.add_argument('--commitment_cost', type=float, default=0.25,
                    help='weight of commitment loss in vector quantizer.')
parser.add_argument('--decay', type=float, default=0.99,
                    help='decay of EMA updates in vector quantizer.')

# Training setting
parser.add_argument('--nr_gpu', type=int, default=2, 
                    help='number of GPUs.')
parser.add_argument('--batch_size', type=int, default=8, 
                    help='batch size in total.')
parser.add_argument('--learning_rate', type=float, default=0.18,
                    help='learning rate.')
parser.add_argument('--dropout_s', type=float, default=0.1, 
                    help='dropout strength. 0 = No dropout, higher = more dropout.')
parser.add_argument('--max_steps', type=int, default=1000000,
                    help='max number of iterations.')
parser.add_argument('--val_steps', type=int, default=10000,
                    help='steps of validation.')
parser.add_argument('--train_spe', type=int, default=10000,
                    help='steps of generating images and saving models.')

# EMA setting
parser.add_argument('--ema_decay', type=float, default=0.9997, 
                    help='decay rate of EMA in validation.')

# Mask setting
parser.add_argument('--random_mask', type=bool, default=False,
                    help='random mask or not.')
parser.add_argument('--mask_size', type=int, default=128,
                    help='provide square masks of this size.')
parser.add_argument('--max_delta', type=int, default=0,
                    help='max delta of masks.')
parser.add_argument('--margins', type=int, default=0,
                    help='margins of masks.')

args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# -----------------------------------------------------------------------------
# Create save folder
folder_name = time.strftime('%Y%m%d-%H%M%S')
folder_name += '_' + args.dataset + '_StructureGenerator'
folder_path = os.path.join(args.checkpoints_dir, folder_name)

if os.path.isdir(args.checkpoints_dir) is False:
    os.mkdir(args.checkpoints_dir)

if os.path.isdir(folder_path) is False:
    os.mkdir(folder_path)

# Data loader
train_loader = DataLoader(flist=args.train_flist, 
                          batch_size=args.batch_size,
                          o_size=args.load_size,
                          im_size=args.image_size,
                          is_train=True)
valid_loader = DataLoader(flist=args.valid_flist, 
                          batch_size=args.batch_size,
                          o_size=args.load_size,
                          im_size=args.image_size,
                          is_train=False)

train_images = train_loader.load_items()
valid_images = valid_loader.load_items()

train_iterator = train_loader.iterator
valid_iterator = valid_loader.iterator

# Generate mask, 1 represents masked point
if args.random_mask:
    bbox = nn.random_bbox(args.image_size, args.image_size, args.margins, args.mask_size, random_mask=True)
    regular_mask = nn.bbox2mask(bbox, args.image_size, args.image_size, args.max_delta, name='mask_c')
    irregular_mask = nn.brush_stroke_mask(args.image_size, args.image_size, name='mask_c')
    mask = tf.cond(tf.less(tf.random_uniform((1,), dtype=tf.float32)[0], 0.5),
                            lambda: regular_mask,
                            lambda: irregular_mask)

else:
    bbox = nn.random_bbox(args.image_size, args.image_size, args.margins, args.mask_size, random_mask=False)
    regular_mask = nn.bbox2mask(bbox, args.image_size, args.image_size, args.max_delta, name='mask_c')
    mask = regular_mask

################### Build structure generator ###################
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

# Split images for distribution training
assert args.batch_size % args.nr_gpu == 0
x_train_list = tf.split(train_images, args.nr_gpu)
x_valid_list = tf.split(valid_images, args.nr_gpu)

# Build model once
batch_pos = x_train_list[0]
enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
dec_gt = vq_decoder(enc_gt['quant_t'], enc_gt['quant_b'], **vq_decoder_opt)
autoencoder_params = []
for v in tf.trainable_variables():
    if 'vector_quantize' not in v.name:
        autoencoder_params.append(v)
ema = tf.train.ExponentialMovingAverage(decay=args.ema_decay)
variables_to_restore = ema.variables_to_restore(moving_avg_variables=autoencoder_params)
# Structure generator
masked = batch_pos * (1. - mask)
cond_masked = structure_condition(masked, mask, ema=None, **structure_condition_opt)
pix_out = structure_pixelcnn(enc_gt['quant_t'], cond_masked, ema=None, dropout_p=args.dropout_s, **structure_pixelcnn_opt) 
structure_generator_params = []
for v in tf.trainable_variables():
    if 'structure' in v.name:
        structure_generator_params.append(v)
maintain_averages_op = tf.group(ema.apply(structure_generator_params))

# Create optimizer
tf_lr = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr)

# Get loss gradients over multiple GPUs
tower_grads = []
loss_train = []
loss_valid = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # Train
        batch_pos = x_train_list[i]
        masked = batch_pos * (1. - mask)
        enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
        encoding_gt = tf.one_hot(enc_gt['idx_t'], args.num_embeddings) 
        # Structure generator
        cond_masked = structure_condition(masked, mask, ema=None, **structure_condition_opt)
        pix_out = structure_pixelcnn(enc_gt['quant_t'], cond_masked, ema=None, dropout_p=args.dropout_s, **structure_pixelcnn_opt)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pix_out, labels=encoding_gt))
        loss_train.append(loss)

        # Gradients
        tower_grads.append(optimizer.compute_gradients(loss, var_list=structure_generator_params))

        # Valid
        batch_pos = x_valid_list[i]
        masked = batch_pos * (1. - mask)
        enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
        encoding_gt = tf.one_hot(enc_gt['idx_t'], args.num_embeddings) 
        # Structure generator
        cond_masked = structure_condition(masked, mask, ema=ema, **structure_condition_opt)
        pix_out = structure_pixelcnn(enc_gt['quant_t'], cond_masked, ema=ema, dropout_p=0., **structure_pixelcnn_opt)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pix_out, labels=encoding_gt))
        loss_valid.append(loss)

# Add losses together and average gradients
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_train[0] += loss_train[i]
        loss_valid[0] += loss_valid[i]        
    # Training op
    grads = nn.average_gradients(tower_grads)
    train_op = tf.group(optimizer.apply_gradients(grads), maintain_averages_op)

# Convert loss to negative log likelihood (bits/dim)
top_shape = (args.image_size//8, args.image_size//8, 1)
bits_per_dim = loss_train[0]/(np.log(2.)*np.prod(top_shape)*args.batch_size)
bits_per_dim_valid = loss_valid[0]/(np.log(2.)*np.prod(top_shape)*args.batch_size)

# Sample structure feature maps
e_sample = tf.placeholder(tf.float32, shape=(args.batch_size//args.nr_gpu, args.image_size//8, args.image_size//8, args.embedding_dim))
h_sample = tf.placeholder(tf.float32, shape=(args.batch_size//args.nr_gpu, args.image_size//8, args.image_size//8, 8*args.nr_channel_cond_s))
batch_pos = x_valid_list[0]
masked = batch_pos * (1. - mask)
img_incomplete = masked
enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
cond_masked = structure_condition(masked, mask, ema=ema, **structure_condition_opt)
pix_out = structure_pixelcnn(e_sample, h_sample, ema=ema, dropout_p=0., **structure_pixelcnn_opt)
pix_out = tf.reshape(pix_out, (-1, args.num_embeddings))
probs_out = tf.nn.log_softmax(pix_out, axis=-1)
samples_out = tf.multinomial(probs_out, 1)
samples_out = tf.reshape(samples_out, (-1, ) + top_shape[:-1])
new_e_gen = tf.nn.embedding_lookup(tf.transpose(enc_gt['embed_t'], [1, 0]), samples_out, validate_indices=False)
# Visualization of generated structure feature maps
dec_gen = vq_decoder(e_sample, tf.zeros_like(enc_gt['quant_b'], dtype=tf.float32), **vq_decoder_opt)
recons_gen = tf.clip_by_value(dec_gen['dec_b'], -1, 1)
# Visualization of ground truth structure feature maps
dec_gt = vq_decoder(enc_gt['quant_t'], tf.zeros_like(enc_gt['quant_b'], dtype=tf.float32), **vq_decoder_opt)
recons_gt = tf.clip_by_value(dec_gt['dec_b'], -1, 1)

# Sample from the model
def sample_from_model(sess):
    gt_np, masked_np, cond_masked_np, recons_gt_np = sess.run([x_valid_list[0], img_incomplete, cond_masked, recons_gt])
    feed_dict = {h_sample: cond_masked_np}
    e_gen = np.zeros((args.batch_size//args.nr_gpu, args.image_size//8, args.image_size//8, args.embedding_dim), dtype=np.float32)
    for yi in range(top_shape[0]):
        for xi in range(top_shape[1]):
            feed_dict.update({e_sample: e_gen})
            new_e_gen_np = sess.run(new_e_gen, feed_dict)
            e_gen[:,yi,xi,:] = new_e_gen_np[:,yi,xi,:]
    recons_gen_np = sess.run(recons_gen, {e_sample: e_gen})
    return gt_np, masked_np, recons_gen_np, recons_gt_np

################### Train structure generator ###################
# Create a saver to restore VQVAE network
restore_saver = tf.train.Saver(variables_to_restore)

# Create a saver to save VQVAE & structure generator
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

# TF session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Initialize dataset
    iterators = [train_iterator.initializer, valid_iterator.initializer]
    sess.run(iterators)
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Restore pre-trained VQVAE network
    ckpt = tf.train.get_checkpoint_state(args.vqvae_network_dir)
    if ckpt and ckpt.model_checkpoint_path:
        restore_saver.restore(sess, ckpt.model_checkpoint_path)
        print('VQVAE network restored ...')
    else:
        print('Restore VQVAE network failed! EXIT!')
        sys.exit()

    # Start to train
    train_nll = []
    begin = time.time()
    for i in range(args.max_steps):
        # Learning rate schedule with linear warm-up and square root decay
        lr = args.learning_rate*args.nr_channel_s**(-0.5)
        lr = lr * min((i + 1)**(-0.35), (i + 1)*16000**(-1.5))
        result = sess.run([train_op, bits_per_dim], {tf_lr:lr})
        train_nll.append(result[1])
        
        # Print training NLL every 100 iterations
        if (i + 1) % 100 == 0:
            print('%d iterations, time: %ds, train NLL: %.5f.' % 
                ((i + 1), time.time()-begin, np.mean(train_nll[-100:])))
            sys.stdout.flush()
            begin = time.time()

        # Validate
        if (i + 1) % args.val_steps == 0:
            # Number of iterations every validation
            # Every iteration will evaluate (num_iter) batches of randomly cropped validation images.
            num_iter = 100

            valid_nll = []
            for step in range(num_iter):
                valid_result = sess.run([bits_per_dim_valid])
                valid_nll.append(valid_result[0])

            # Print validation NLL
            print('%d iterations, time: %ds, valid NLL: %.5f.' % 
                ((i + 1), time.time()-begin, np.mean(valid_nll)))
            sys.stdout.flush()
            begin = time.time()

        # Generate structure feature maps & Save model
        if (i + 1) % args.train_spe == 0:
            # Generate structure feature maps
            gt_np, masked_np, recons_gen_np, recons_gt_np = sample_from_model(sess)
            nn.structure_visual(gt_np, masked_np, recons_gen_np, recons_gt_np, (i + 1), args.image_size, folder_path)
            # Print generation time
            print('%d iterations, generation time: %.3fs.' % ((i + 1), time.time()-begin))
            sys.stdout.flush()

            # Save model
            checkpoint_path = os.path.join(folder_path, 'model.ckpt')
            saver.save(sess, checkpoint_path)
            begin = time.time()
