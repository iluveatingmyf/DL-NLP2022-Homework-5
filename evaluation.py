import numpy as np
import tensorflow.compat.v1 as tf
from seq_loss import sequence_loss_by_example
tf.compat.v1.disable_eager_execution()
from train import RNNModel
import re
import jieba

#字符是否为乱码
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——'):
            return True
    return False



def load_trainset(data):
    # 利用正则规则进行数据预处理，替换特殊符号
    #
    pattern = re.compile(r'\(.*\)')
    data = [pattern.sub('', lines) for lines in data]
    data = [line.replace('……', '。') for line in data if len(line) > 1]
    data = ''.join(data)
    data = [char for char in data if is_uchar(char)]
    data = ''.join(data)
    word_data = list(jieba.cut(data))
    return word_data

def vector_generate(word_data):
    word_vocab = set(word_data)
    id2word = list(word_vocab)
    word2id = {c: i for i, c in enumerate(word_vocab)}
    return word2id,id2word



tf.compat.v1.reset_default_graph()
#生成向量字典
with open('./dataset/天龙八部.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    # 数据预处理与分词
    word_data = load_trainset(data)
    # 生成词语与one-hot向量的映射向量
    word2id,id2word = vector_generate(word_data)

#替换测试数据
test = '他若是你弟子，碍着你的面子，我也不能做得太绝了，既是寻常宾客，那可不能客气了。有人竟敢在剑湖宫中讥笑‘无量剑’东宗的武功'
test_word_list = load_trainset(data)
# 转换数据为数字格式
numdata = [word2id[word] for word in test_word_list]



VOCAB_SIZE = 52575
HIDDEN_SIZE = 512
HIDDEN_LAYERS = 3
MAX_GRAD_NORM = 1
learning_rate = 0.003
TIME_STEPS = 100
BATCH_SIZE = 8
BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)

evalmodel = RNNModel(1, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE,TIME_STEPS,BATCH_NUMS, learning_rate)

print(evalmodel)
# 加载模型
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './checkpoints/lstm.ckpt')
    new_state = sess.run(evalmodel.initial_state)
    samples = []
    x = np.array([numdata])
    for i in range(100):
        print(i)
        feed = {evalmodel.inputs: x, evalmodel.keepprb: 1., evalmodel.initial_state: new_state}
        c, new_state = sess.run([evalmodel.predict, evalmodel.final_state], feed_dict=feed)
        for j in range(len(numdata)):
            x[0][j] = c[0]
        samples.append(c[0])
    print('test:', ''.join([id2word[index] for index in samples]))
