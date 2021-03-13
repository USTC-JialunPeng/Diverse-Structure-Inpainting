import os
import numpy as np
import tensorflow as tf

class DataLoader():
    """Construct dataset class."""
    def __init__(self, flist, batch_size, o_size, im_size, is_train):
        self.flist = self.load_flist(flist)
        self.batch_size = batch_size
        self.o_size = o_size
        self.im_size = im_size
        self.is_train = is_train
        self.iterator = None

    def __len__(self):
        """Get the length of dataset."""
        return len(self.flist)

    def load_items(self):
        images = self.load_images()
        return images

    def input_parse(self, img_path):
        img_file = tf.read_file(img_path)
        img_decoded = tf.cond(tf.image.is_jpeg(img_file),
                            lambda: tf.image.decode_jpeg(img_file, channels=3),
                            lambda: tf.image.decode_png(img_file, channels=3))
        img = tf.cast(img_decoded, tf.float32)

        if self.is_train:
            img = tf.image.resize_images(img, [self.o_size, self.o_size], method=0)
            img = tf.image.random_crop(img, [self.im_size, self.im_size, 3])
        else:
            img = tf.image.resize_images(img, [self.im_size, self.im_size], method=0)
            
        img = tf.clip_by_value(img, 0., 255.)
        img = img / 127.5 - 1

        return img

    def load_images(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.flist)
        dataset = dataset.map(self.input_parse)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat()

        self.iterator = dataset.make_initializable_iterator()
        images = self.iterator.get_next()

        return images

    def load_images_once(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.flist)
        dataset = dataset.map(self.input_parse)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat(1)

        self.iterator = dataset.make_initializable_iterator()
        images = self.iterator.get_next()

        return images

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + \
                    list(glob.glob(flist + '/*.JPG'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []