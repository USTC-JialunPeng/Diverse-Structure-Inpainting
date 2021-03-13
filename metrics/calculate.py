import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import skimage.measure
import inception_score
import fid


parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir', type=str, default='/gdata/result/celeba_hq', 
                    help='gen are saved here.')
parser.add_argument('--gt_dir', type=str, default='/gdata/test_set/celeba_hq', 
                    help='gt are saved here.')
parser.add_argument('--test_num', default=1000, type=int, 
                    help='number of test images.')
args = parser.parse_args()

# Create the FID_inception graph for FID metric
inception_path = fid.check_or_download_inception()
fid.create_inception_graph(inception_path)

# sess
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    psnr_list = []
    ssim_list = []
    gen_list = []
    gen_exp_list = []
    gt_exp_list = []
    for i in range(args.test_num):
        im1 = cv2.imread(os.path.join(args.gen_dir, '%05d.png' % i))[:,:,::-1]
        im2 = cv2.imread(os.path.join(args.gt_dir, 'img_%05d.png' % i))[:,:,::-1]
        psnr_list.append(skimage.measure.compare_psnr(im1, im2, 255))
        ssim_list.append(skimage.measure.compare_ssim(im1, im2, multichannel=True))
        # IS & FID
        gen_list.append(im1)
        gen_exp_list.append(np.expand_dims(im1, axis=0))
        gt_exp_list.append(np.expand_dims(im2, axis=0))
    gen_arr = np.concatenate(gen_exp_list, axis=0)
    gt_arr = np.concatenate(gt_exp_list, axis=0)

    # IS & MIS metric
    mean1, mean2 = inception_score.get_inception_score(gen_list, sess)

    # FID metric
    mu_gen, sigma_gen = fid.calculate_activation_statistics(gen_arr, sess, batch_size=50)
    mu_gt, sigma_gt = fid.calculate_activation_statistics(gt_arr, sess, batch_size=50)
    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

    print('PSNR: %.5f' % (np.mean(psnr_list)))
    print('MS-SSIM: %.5f.' % (np.mean(ssim_list)))
    print('IS: %.5f.' % mean1)
    print('MIS: %.5f.' % mean2)
    print('FID: %.5f.' % fid_value)
