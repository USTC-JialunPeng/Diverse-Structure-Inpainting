import os.path
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import math

MODEL_DIR = '/gdata/inception_model'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def get_inception_score(images, sess, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 1
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'IS_Inception_Net/ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    IS = []
    MIS = []

    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits): (k+1) * (preds.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        scores1 = []
        scores2 = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores1.append(entropy(pyx, py))
            scores2.append(-entropy(pyx))
        IS.append(np.exp(np.mean(scores1)))
        MIS.append(np.exp(np.mean(scores2)))

    return np.mean(IS), np.mean(MIS)

# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='IS_Inception_Net')

    # Works with an arbitrary minibatch size.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        pool3 = sess.graph.get_tensor_by_name('IS_Inception_Net/pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("IS_Inception_Net/softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)

if softmax is None:
    _init_inception()
