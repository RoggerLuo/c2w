import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import timeline

from get_feed_fn import get_feed_fn
from logging_hook import get_logging_hook

model_dir='model/seq2seq'

def train(est,vocab):

    def input_fn():
        
        inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
        output = tf.placeholder(tf.int64, shape=[None, None], name='output') # output 指label吗
        
        # 创建placeholder 然后切片放进打印机打印
        tf.identity(inp[0], 'input_0') # 好像内存中多了一个叫做input_0的op，然后到时候就可以根据名字调用
        tf.identity(output[0], 'output_0')
        
        return {'input': inp,'output': output}, None # 这个None是怎么回事


    # Make hooks to print examples of inputs/predictions.
    print_inputs = get_logging_hook(['input_0', 'output_0'],vocab) 
    print_predictions = get_logging_hook(['predictions', 'train_pred'],vocab) # predictions和train_pred是啥 不一样吗
    timeline_hook = timeline.TimelineHook(model_dir, every_n_iter=100)

    est.train(
        input_fn=input_fn,
        hooks=[ # 4个hook
            tf.train.FeedFnHook(get_feed_fn(vocab)), 
            print_inputs, 
            print_predictions,
            timeline_hook
        ], 
        steps=10000
    )

