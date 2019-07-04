import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt

TIME_STEPS=28
BATCH_SIZE=128
HIDDEN_UNITS1=30
HIDDEN_UNITS=10
LEARNING_RATE=0.001
EPOCH=50


TRAIN_EXAMPLES=42000
TEST_EXAMPLES=28000


#------------------------------------Generate Data-----------------------------------------------#
#generate data

train_frame = pd.read_csv("../Mnist/train.csv") 
test_frame = pd.read_csv("../Mnist/test.csv") # pop the labels and one-hot
train_labels_frame = train_frame.pop("label")

X_train = train_frame.astype(np.float32).values 
y_train=pd.get_dummies(data=train_labels_frame).values 
X_test = test_frame.astype(np.float32).values


X_train=np.reshape(X_train,newshape=(-1,28,28))
X_test=np.reshape(X_test,newshape=(-1,28,28))

graph=tf.Graph()
with graph.as_default():
     X_p=tf.placeholder(dtype=tf.float32,shape=(None,TIME_STEPS,28),name="input_placeholder") 
     y_p=tf.placeholder(dtype=tf.float32,shape=(None,10),name="pred_placeholder")
     lstm_forward_1=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1) 
     lstm_forward_2=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS) 
     lstm_forward=rnn.MultiRNNCell(cells=[lstm_forward_1,lstm_forward_2])
     lstm_backward_1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1) 
     lstm_backward_2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS) 
     lstm_backward=rnn.MultiRNNCell(cells=[lstm_backward_1,lstm_backward_2])
     
     outputs,states=tf.nn.bidirectional_dynamic_rnn( cell_fw=lstm_forward, cell_bw=lstm_backward, inputs=X_p, dtype=tf.float32 )
     outputs_fw=outputs[0]
     outputs_bw = outputs[1]
     h=outputs_fw[:,-1,:]+outputs_bw[:,-1,:]
     cross_loss=tf.losses.softmax_cross_entropy(onehot_labels=y_p,logits=h) #print(loss.shape)
     correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
     optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=cross_loss) 
     init=tf.global_variables_initializer()
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1,EPOCH+1):
         train_losses=[] 
         accus=[] #test_losses=[] 
         print("epoch:",epoch) 
         for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
              _,train_loss,accu=sess.run( fetches=(optimizer,cross_loss,accuracy), 
                                         feed_dict={ X_p:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                                                    y_p:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE] } )
              train_losses.append(train_loss)
              accus.append(accu)
         print("average training loss:", sum(train_losses) / len(train_losses))
         print("accuracy:",sum(accus)/len(accus))
--------------------- 
作者：luoganttcc 
来源：CSDN 
原文：https://blog.csdn.net/luoganttcc/article/details/83384823 
版权声明：本文为博主原创文章，转载请附上博文链接！
