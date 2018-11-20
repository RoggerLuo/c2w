import tensorflow as tf
import numpy as np


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def bidirectional_encoder(FLAGS,inputs,keep_prob):
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]

    print('FLAGS.time_step:(这个time_step需要和batch中每个句子的字符长度一致)')
    print(FLAGS.time_step)

    
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
    print('inputs length after unstack:')
    print(len(inputs))
    print('inputs[0].shape length after unstack:')
    print(inputs[0].shape)

    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)

    print('output length (== time_step):')
    print(len(output))

    print('FLAGS.num_units:')
    print(FLAGS.num_units)

    print('output[0].shape ( == double num_units):')
    print(output[0].shape)

    output = tf.stack(output, axis=1)
    return output
    """ output
    Tensor("stack:0", shape=(?, 32, 100), dtype=float32)
    (?是batch_size，32是timestep，100是units_num)
    """

def encoder(FLAGS,inputs,keep_prob):
    cell_fw = [lstm_cell(FLAGS.num_units*2, keep_prob) for _ in range(FLAGS.num_layer)]
    print('FLAGS.time_step:(这个time_step需要和batch中每个句子的字符长度一致)')
    print(FLAGS.time_step)
    print(inputs)

    print('inputs[0].shape length after unstack:')
    print(inputs[0].shape)
    '''
        [max_size,batch_size,embed_size]
    '''
    encoder = tf.contrib.rnn.MultiRNNCell(cell_fw)
    output, _ = tf.nn.dynamic_rnn(encoder, inputs, dtype=tf.float32)

    print('FLAGS.num_units:')
    print(FLAGS.num_units)

    print('output[0].shape ( == double num_units):')
    print(output[0].shape)
    return output
    """ output是一个tensor
    <tf.Tensor 'rnn/transpose:0' shape=(?, 32, 100) dtype=float32>
    """
