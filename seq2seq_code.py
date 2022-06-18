import re
import jieba
import numpy as np
import tensorflow.compat.v1 as tf
from seq_loss import sequence_loss_by_example
tf.compat.v1.disable_eager_execution()

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
    return id2word,word2id


def data_generator(data, batch_size, time_steps):
    samples_per_batch = batch_size * time_steps
    batch_nums = len(data) // samples_per_batch
    data = data[:batch_nums*samples_per_batch]
    data = data.reshape((batch_size, batch_nums, time_steps))
    for i in range(batch_nums):
        x = data[:, i, :]
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        try:
            y[:, -1] = data[:, i+1, 0]
        except:
            y[:, -1] = data[:, 0, 0]
            yield x, y


#搭建模型
class RNNModel():
    """docstring for RNNModel"""
    def __init__(self, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate):
        super(RNNModel, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.HIDDEN_LAYERS = HIDDEN_LAYERS
        self.VOCAB_SIZE = VOCAB_SIZE

        # 定义占位符
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None])
            self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, None])
            self.keepprb = tf.placeholder(tf.float32)

        # 定义词嵌入层
        with tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
            emb_input = tf.nn.embedding_lookup(embedding, self.inputs)
            emb_input = tf.nn.dropout(emb_input,  rate = 1-self.keepprb)

        # 搭建lstm结构
        with tf.name_scope('rnn'):
            lstm = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keepprb)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * HIDDEN_LAYERS)
            self.initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell, emb_input, initial_state=self.initial_state)

        # 重新reshape输出
        with tf.name_scope('output_layer'):
            outputs = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
            w = tf.get_variable('outputs_weight', [HIDDEN_SIZE, VOCAB_SIZE])
            b = tf.get_variable('outputs_bias', [VOCAB_SIZE])
            logits = tf.matmul(outputs, w) + b

        # 计算损失
        with tf.name_scope('loss'):
            self.loss = sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])], [tf.ones([BATCH_SIZE * TIME_STEPS], dtype=tf.float32)])
            self.cost = tf.reduce_sum(self.loss) / BATCH_SIZE

        # 优化算法
        with tf.name_scope('opt'):
            # 学习率衰减
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(tf.cast(learning_rate,tf.float32), global_step, BATCH_NUMS, 0.99, staircase=True)



            #通过clip_by_global_norm()控制梯度大小
            trainable_variables = tf.trainable_variables()

            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), 1)
            self.opt = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, trainable_variables))

        # 预测输出
        with tf.name_scope('predict'):
            self.predict = tf.argmax(logits, 1)


if __name__ =="__main__":
    #读取原始数据
    with open('./dataset/天龙八部.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        #数据预处理与分词
        word_data = load_trainset(data)
        #生成词语与one-hot向量的映射向量
        id2word, word2id = vector_generate(word_data)

        #转换数据为数字格式
        numdata = [word2id[word] for word in word_data]
        numdata = np.array(numdata)

        # 预定义模型参数
        VOCAB_SIZE = len(set(word_data))
        EPOCHS = 1000
        BATCH_SIZE = 8
        TIME_STEPS = 100
        BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)
        HIDDEN_SIZE = 512
        HIDDEN_LAYERS = 6
        MAX_GRAD_NORM = 1
        learning_rate = 0.05

        # 训练模型
        model = RNNModel(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE,TIME_STEPS)
        print(model)

        # 保存模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())

            sess.run(tf.global_variables_initializer())
            for k in range(EPOCHS):
                state = sess.run(model.initial_state)
                train_data = data_generator(numdata, BATCH_SIZE, TIME_STEPS)
                total_loss = 0.
                for i in range(BATCH_NUMS):
                    try:
                        xs, ys = next(train_data)
                        feed = {model.inputs: xs, model.targets: ys, model.keepprb: 0.8, model.initial_state: state}
                        costs, state, _ = sess.run([model.cost, model.final_state, model.opt], feed_dict=feed)
                        total_loss += costs
                        if (i + 1) % 50 == 0:
                            print('epochs:', k + 1, 'iter:', i + 1, 'cost:', total_loss / i + 1)
                    except StopIteration:
                        break
            saver.save(sess, './checkpoints/lstm.ckpt')

        writer.close()


        # ============模型测试============
        with open('./test_text.txt', 'r', encoding='utf-8') as f:
            test_string = f.readlines()
            print(f)
            T = []
            test_process_string =load_trainset(test_string)
            print(test_process_string)
            for word in test_process_string:
                tmp = word2id[word]
                T.append(tmp)
            print(T)
            print(np.array([T]))

            tf.reset_default_graph()
            evalmodel = RNNModel(1, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate)
            # 加载模型
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, './checkpoints/lstm.ckpt')
                new_state = sess.run(evalmodel.initial_state)
                x = np.array([T])
                samples = []
                for i in range(100):
                    feed = {evalmodel.inputs: x, evalmodel.keepprb: 1., evalmodel.initial_state: new_state}
                    c, new_state = sess.run([evalmodel.predict, evalmodel.final_state], feed_dict=feed)
                    for j in range(len(T)):
                        x[0][j] = c[0]
                        samples.append(c[0])
                print(x)
                print(len(c))
                print(samples)
                print('test:', ''.join([id2word[index] for index in samples]))
