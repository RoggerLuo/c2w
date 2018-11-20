# Char-language model

### Nov 20
`dynamic_rnn`和`stack_bidirectional_rnn`的返回是一样的  
**是一个tensor， 不是list**


### Nov 19
项目中要用绝对路径

不是把c2w做成lm  
c2w本质是bi  
只是为了借用bi  把bi拿出来  
放到seq2seq里去

c2w(原bi)其实是一个分类，  
可以当成mlp


### Nov 19
### `dynamic_rnn`和`stack_bidirectional_rnn`
``` python
cell_fw = [lstm_cell(FLAGS.num_units*2, keep_prob) for _ in range(FLAGS.num_layer)]
encoder = tf.contrib.rnn.MultiRNNCell(cell_fw)
output, _ = tf.nn.dynamic_rnn(encoder, inputs, dtype=tf.float32)
```
`bidirectional_rnn`unstack和重新stack之后，  
和`dynamic_rnn`接口数据的结构就一模一样了  

另外，bidirection cell的units_num也是数量也是翻倍的


`stack_bidirectional_rnn`相当于  
`MultiRNNCell` + `dynamic_rnn`

``` python
cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]

inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
output = tf.stack(output, axis=1)

```

