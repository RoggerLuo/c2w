import numpy as np
import tensorflow as tf
import config
import pkl

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

input_max_length,output_max_length = config.get_max()
batch_size = config.get_size()

sentence_list = pkl.read('./corpus.pkl')
''' sentence_list: 
[
    'sentence 1',
    'sentence 2',
    'sentence 3',
    ...
]
'''
def get_feed_fn(vocab): # input_max_length
    """ vocab: (就是word2id)
    {'<S>': 0, '</S>': 1, '<UNK>': 2, '，': 3, '的': 4, '。': 5, '、': 6, '０': 7, '一': 8, '在': 9, '行': 10 }
    """

    input_filename='input'
    output_filename='output'
    def word2id(sentence):        
        return [vocab.get(word, UNK_TOKEN) for word in sentence]

    def sampler():
        for sentence in sentence_list: 
            
            if len(sentence) < 4: # 太短的句子就不要了
                continue
            
            sentence_in_id = word2id(sentence)
            """sentence_in_id: 
            [14, 574, 123, 1294, 265, 38...]
            """

            # 裁剪
            input_ = sentence_in_id[:input_max_length - 1] + [END_TOKEN] 
            # 结尾出现end_token的位置，不能乱来，不然会出问题，可能是因为它是以end_token来判断input是否结束的
            yield {
                'input': input_,
                'output': input_
            }
            """
            'input':[14, 574, 123, 1294, 265, 38,
            """

    sample_me = sampler() # idx的输入
    # return sample_me
    """
    FeedFnHook只做了一件事
    继承SessionRunHook,然后实现一个方法：
    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(fetches=None, feed_dict=self.feed_fn()) # 注意这里
    """
    # 每次执行都给input output placeholder喂学习数据，batch size的数据
    # batch size在这里拼装, 
    def feed_fn(): # 每次sess.run都会重新执行feed_fn()
        inputs, outputs = [], []
        input_length, output_length = 0, 0
        for i in range(batch_size):
            res = next(sample_me)
            inputs.append(res['input']) # 把单个的素材 推进list
            outputs.append(res['output'])
            input_length = max(input_length, len(inputs[-1])) # inputs[-1]，inputs的最后一个，就是刚才append上的那一个 
            output_length = max(output_length, len(outputs[-1])) # 取batch中的最长值
        # Pad me right with </S> token.
        for i in range(batch_size):
            # 长度不够的地方填充end_token
            inputs[i] += [END_TOKEN] * (input_length - len(inputs[i])) # [7] * 5 = [7,7,7,7,7]
            outputs[i] += [END_TOKEN] * (output_length - len(outputs[i])) # [7,7,7,7,7] + [1,2,3] = [7 7 7 7 7 1 2 3]

        return {'input:0': inputs,'output:0': outputs} # return dict of tensor (feed_dict)
    return feed_fn
