import numpy as np
import tensorflow as tf

def decode(helper, scope, output_max_length,reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        cell = tf.contrib.rnn.GRUCell(num_units=num_units)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=num_units, memory=encoder_outputs,memory_sequence_length=input_lengths)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=num_units / 2)
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            attn_cell, vocab_size, reuse=reuse
        )
        
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell, vocab_size, reuse=reuse
        )
        
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=out_cell.zero_state( # use_attention 
                dtype=tf.float32, batch_size=batch_size))
        
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, 
            output_time_major=False,
            impute_finished=True, 
            maximum_iterations=output_max_length
        )
        return outputs[0]