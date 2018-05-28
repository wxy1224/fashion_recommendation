# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu

import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized


@graph_memoized
def get_dorefa(bitW, bitA, bitG):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x):
        if bitW == 32:
            return x
        if bitW == 1:   # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    @tf.RegisterGradient("FGGrad")
    def grad_fg(op, x):
        rank = x.get_shape().ndims
        assert rank is not None
        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        n = float(2**bitG - 1)
        x = x * 0.5 + 0.5 + tf.random_uniform(
            tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = quantize(x, bitG) - 0.5
        return x * maxx * 2

    def fg(x):
        if bitG == 32:
            return x
        with G.gradient_override_map({"Identity": "FGGrad"}):
            return tf.identity(x)
    return fw, fa, fg


def ternarize(x, thresh=0.05):
    """
    Implemented Trained Ternary Quantization:
    https://arxiv.org/abs/1612.01064

    Code modified from the authors' at:
    https://github.com/czhu95/ternarynet/blob/master/examples/Ternary-Net/ternary.py
    """
    G = tf.get_default_graph()
    shape = x.get_shape()

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thresh)

    w_p = tf.get_variable('Wp', initializer=1.0, dtype=tf.float32)
    w_n = tf.get_variable('Wn', initializer=1.0, dtype=tf.float32)

    tf.summary.scalar(w_p.op.name + '-summary', w_p)
    tf.summary.scalar(w_n.op.name + '-summary', w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w = tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np

    tf.summary.histogram(w.name, w)
    return w
