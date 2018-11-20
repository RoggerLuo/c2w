import argparse
import tensorflow as tf
import math
import numpy as np
from os.path import join
from getEncoder import bidirectional_encoder,encoder

def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)

def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)

def getY(FLAGS,x):        
    print('FLAGS.vocab_size:')
    print(FLAGS.vocab_size)
    print('FLAGS.embedding_size:')
    print(FLAGS.embedding_size)

    # Embedding Layer
    with tf.variable_scope('embedding'):
        embedding = tf.Variable(tf.random_normal([FLAGS.vocab_size, FLAGS.embedding_size]), dtype=tf.float32)
        print('embedding', embedding)

    inputs = tf.nn.embedding_lookup(embedding, x)
    print('inputs.shape:')
    print(inputs.shape)
    
    # FLAGS.keep_prob
    keep_prob = tf.placeholder(tf.float32, [])

    # RNN Layer
    output=encoder(FLAGS,inputs,keep_prob)
    # output=bidirectional_encoder(FLAGS,inputs,keep_prob)

    output = tf.reshape(output, [-1, FLAGS.num_units * 2])
    print('output after Reshape')
    print(output.shape)

    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])

        # w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        # b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b

    print('FLAGS.category_num')
    print(FLAGS.category_num)

    print('y.shape')
    print(y.shape)
    return y,keep_prob
