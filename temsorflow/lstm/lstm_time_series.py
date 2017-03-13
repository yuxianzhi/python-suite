#!/usr/bin/python
# -*- coding: utf-8 -*
#tensorflow version 1.0.0


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys
#import requests
#import io


 
#load data
#url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'  
#ass_data = requests.get(url).content
#df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  #python2 use StringIO.StringIO
f=open('data.csv')
df=pd.read_csv(f)
 
data = np.array(df['renshu'])
#print data
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)
 
seq_size = 7
train_x, train_y = [], []
for i in range(len(normalized_data) - seq_size - 1):
	train_x.append(np.expand_dims(normalized_data[i : i + seq_size], axis=1).tolist())
	train_y.append(normalized_data[i + 1 : i + seq_size + 1].tolist())
 
input_dim = 1
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])
Y = tf.placeholder(tf.float32, [None, seq_size])
 
# regression
def ass_rnn(hidden_layer_size=6):
	W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
	b = tf.Variable(tf.random_normal([1]), name='b')
	cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
	outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
	W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
	out = tf.matmul(outputs, W_repeated) + b
	out = tf.squeeze(out)
	return out


model_path='./ass.model'
out = ass_rnn()

def train_rnn():
#	with tf.variable_scope('train_lstm'):
#		out = ass_rnn()
 
	loss = tf.reduce_mean(tf.square(out - Y))
	train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)
 
	# Initializing the variables  
	init = tf.initialize_all_variables()
	# 'Saver' op to save and restore all the variables  
	saver = tf.train.Saver()
	#saver = tf.train.Saver(tf.global_variables())
	#saver = tf.train.Saver(tf.all_variables())
	with tf.Session() as sess:
		#tf.get_variable_scope().reuse_variables()
		#sess.run(tf.global_variables_initializer())
 		#sess.run(tf.initialize_all_variables())
		sess.run(init)

		for step in range(10000):
			_, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
#			if step % 10 == 0:
#				#asses model loss
#				print(step, loss_)
		print("save model: ", saver.save(sess, model_path))
#		all_vars =tf.all_variables()#tf.get_collection('vars')
#		for v in all_vars:
#    			v_ = sess.run(v)
#    			print(v,v_)

train_rnn()
 



def array_out(array):
        dianjia=sys.argv[1]

        f = file('predict.csv','aw')
        f.write(dianjia)
        for i in array:
                f.write(",")
                f.write(str(int(max(0,round(i*np.std(data)+np.mean(data))))))
     	f.write('\n')
        f.close()


def prediction():
#	with tf.variable_scope('prediction'):
#		out = ass_rnn()
         # Initializing the variables  
        init = tf.initialize_all_variables()
        # 'Saver' op to save and restore all the variables  
        saver = tf.train.Saver()

	#saver = tf.train.Saver(tf.global_variables())
	#saver=tf.train.import_meta_graph('ass.model.meta')
	with tf.Session() as sess:
		#init_op = tf.initialize_all_variables()
		#sess.run(init_op)
		#sess.run(tf.global_variables_initializer())
		sess.run(init)
		#module_file = tf.train.latest_checkpoint('./')
		#tf.get_variable_scope().reuse_variables()
		saver.restore(sess, model_path)
#                all_vars =tf.all_variables()#tf.get_collection('vars')
#                for v in all_vars:
#                        v_ = sess.run(v)
#                        print(v,v_)

		prev_seq = train_x[-1]
		predict = []
		for i in range(14):
			next_seq = sess.run(out, feed_dict={X: [prev_seq]})
			predict.append(next_seq[-1])
			prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

		array_out(predict)
		#fileSave1 = pd.DataFrame({'normalized_data':normalized_data})  
		#fileSave1.to_csv('normalized_data.csv',index=False,sep=',') 
                #fileSave1 = pd.DataFrame({'predict':predict})
                #fileSave1.to_csv('predict.csv',index=False,sep=',')
#                print normalized_data
#                print predict
#		plt.figure()
#		plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
#		plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
#		plt.show()
 
prediction()
