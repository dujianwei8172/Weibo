learning_rate = 0.1
state_size = 128 # hidden layer num of features
n_classes = 19
n_features = 23

#输入
x = tf.placeholder(tf.float32, [None, None, n_features], name='input_placeholder')  #batch_size, time_step, feat_len
y = tf.placeholder(tf.float32, [None, None, n_classes], name='labels_placeholder')  #batch_size, time_step, n_classes

batch_size = tf.placeholder(tf.int32, (), name='batch_size')
time_steps = tf.placeholder(tf.int32, (), name='times_step')

#双向rnn
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)

init_fw = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
init_bw = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)

weights = tf.get_variable("weights", [2 * state_size, n_classes], dtype=tf.float32,   #注意这里的维度
                         initializer = tf.random_normal_initializer(mean=0, stddev=1))
biases = tf.get_variable("biases", [n_classes], dtype=tf.float32, 
                        initializer = tf.random_normal_initializer(mean=0, stddev=1))

outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                       lstm_bw_cell,
                                                       x,
                                                       initial_state_fw = init_fw,
                                                       initial_state_bw = init_bw)

outputs = tf.concat(outputs, 2)   #将前向和后向的状态连接起来
state_out = tf.matmul(tf.reshape(outputs, [-1, 2 * state_size]), weights) + biases  #注意这里的维度
logits = tf.reshape(state_out, [batch_size, time_steps, n_classes])

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y))  #计算交叉熵
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  #优化方法
train_op = optimizer.minimize(loss_op)

#进行softmax计算
probs = tf.nn.softmax(logits, -1)   #-1也是默认值，表示在最后一维进行运算
predict = tf.argmax(probs, -1)#最大的概率在最后一维的哪一列，从0计数
                                #维度变为  batch_size * time_step

def train_network(num_epochs = 100):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  #初始化variable
        
        intervel = 5
        
        for epoch in range(num_epochs):
            #开始训练
            for idx, (time_step, inputs, labels, idxes) in enumerate(get_dump_seq_data(1)):
                _= sess.run([train_op],
                           feed_dict = {x: inputs,
                                       y:labels,
                                       batch_size:len(inputs),
                                       time_steps: time_step})
            print("epoch %d train done" % epoch)
            #这一轮训练完毕，计算损失值和准确率
            
            if epoch % intervel == 0 and epoch > 1:
                #训练集误差
                acc_record, total_df, total_acc, loss = compute_accuracy(sess, 1)  #这里是我自定义的函数，与整个架构关系不大
                #验证集误差
                acc_record_dev, total_df_dev, total_acc_dev, loss_dev = compute_accuracy(sess, 0)
                print("train_acc: %.6f, train_loss: %.6f; dev_acc: %.6f, dev_loss: %.6f" % (total_acc, loss, total_acc_dev, loss_dev))
                print("- " * 50)
                if num_epochs - epoch <= intervel:
                    return acc_record, total_df, acc_record_dev, total_df_dev
