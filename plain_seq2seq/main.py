from seq2seq import seq2seq
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import timeline

from logging_hook import get_formatter
import config
import pkl
import logging # 如果不加logging

from est_train import train

tf.logging._logger.setLevel(logging.INFO) # 和tf.logging，logging_hook信息就打印不出来

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

input_max_length,output_max_length = config.get_max()
model_dir='model/seq2seq'
from get_feed_fn import get_feed_fn

def load_vocab(filename):
    pklData = pkl.read(filename) # list ['我','是']
    vocab = {}
    for idx, item in enumerate(pklData):
        vocab[item.strip()] = idx
    return vocab 
    """ vocab: (就是word2id)
    {'<S>': 0, '</S>': 1, '<UNK>': 2, '，': 3, '的': 4, '。': 5, '、': 6, '０': 7, '一': 8, '在': 9, '行': 10 }
    """

def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}

def get_formatter(vocab): # 工厂函数 name_list应该说是operation name list
    rev_vocab = get_rev_vocab(vocab)

    def to_str(sequence): # tensor的值是一个一维list
        tokens = [rev_vocab.get(x, "<UNK>") for x in sequence] # 如果没有x就UNK
        return ' '.join(tokens)

    def formatter(sentences): # dict of tag->tensor, dict就是js中的对象,tag是key，tensor是值
        res = []

        for sentence in sentences:
            res.append(to_str(sentence))
        return '\n'.join(res)

        # for name in name_list:
        #     res.append("%s = %s" % (name, to_str(values[name])))
        # return '\n'.join(res)
    return formatter 

if __name__ == "__main__":
    vocab = load_vocab('./vocab.pkl') 
    formatter = get_formatter(vocab)

    """ vocab: (就是word2id)
    {'<S>': 0, '</S>': 1, '<UNK>': 2, '，': 3, '的': 4, '。': 5, '、': 6, '０': 7, '一': 8, '在': 9, '行': 10 }
    """
    def input_fn():
        inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
        output = tf.placeholder(tf.int64, shape=[None, None], name='output') # output 指label吗
        
        # 创建placeholder 然后切片放进打印机打印
        tf.identity(inp[0], 'input_0') # 好像内存中多了一个叫做input_0的op，然后到时候就可以根据名字调用
        tf.identity(output[0], 'output_0')
        
        return {'input': inp,'output': output}#, None # 这个None是怎么回事


    params = {
        'vocab_size': len(vocab),
        'embed_dim': 100, # embed_dim和num_units可以不同？
        'num_units': 256
    }    
    # 'batch_size': 32,

    mode,loss,train_op,predictions = seq2seq(mode='mode',features=input_fn(), labels={}, params=params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # 每次不写就会报错

    feed_fn = get_feed_fn(vocab)
    for i in range(500):
        predictions_, loss_, _ = sess.run([predictions,loss,train_op],feed_dict=feed_fn())
        print(formatter(predictions_))
        print(loss_)

    # print(train_op)

    # est = tf.estimator.Estimator(
    #     model_fn=seq2seq,
    #     model_dir=model_dir, 
    #     params=params
    # )
    # train(est,vocab)
    # predict(est)


def predict(est):
    """
    因为把output当成feature传入，而不是labels
    所以numpy_input_fn中y不填
# 
    predict的时候output用不到，但是必须填,
    因为用predict和train用的同一个模型，
    
    如果没有output，执行模型的时候会报错,
    output的shape不对，也会报错

    报错很奇怪
    Length of tensors in x and y is mismatched

    """
    input_ = np.array([[1,2],[3,4]])
    output_ = np.array([[100,2],[3,4]]) # 要符合seq2seq模型中tf.concat的要求，shape必须是(batch_size,x_length)
    x={'input':input_,'output':output_}
    
    """
    把所有的数据变成一个巨大的batch传进入x，它会自己划分成更小的batch
    假设原作者手动拼出来的batch数据,size为2,（即t_data.input_fn这个函数所生成的batch数据）是 
        [
            item1,
            item2
        ]
    那么传入的格式是 
        [
            item,
            item,
            item,
            item,
            ...
        ]
    如果设置的batch_size为3
    那么，feature['input']将会得到
        [item,item,item]
    每次执行next，都会拿取一个新的batch，如果list中的数据全部用完了，那就会重新从头部开始拿取
    """
    input_fn = tf.estimator.inputs.numpy_input_fn(x, y=None, batch_size=1, shuffle=False, num_epochs=10)
    predictions = est.predict(input_fn=input_fn)
    print(predictions)
    print(next(predictions))
    print(next(predictions))
    # 再往后，没有数据了，batch_size就只能循环以前的数据，
    print(next(predictions))
    print(next(predictions))
    print(next(predictions))
