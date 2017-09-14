import tensorflow as tf
import numpy as np
import os

batch_size = 256
sequence_length = 20
num_encoder_symbols = 50002  # 'UNK' and '_'
num_decoder_symbols = 50002
embedding_size = 512
learning_rate = 0.001
model_path = './model/model.ckpt'

encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

logits = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length, num_decoder_symbols])
targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])
train_weights = np.ones(shape=[batch_size, sequence_length], dtype=np.float32)
cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length)


def loadQA():
    train_x = np.load('./data/idx_q.npy', mmap_mode='r')
    train_y = np.load('./data/idx_a.npy', mmap_mode='r')
    return train_x, train_y

encoder_inputs = tf.unstack(encoder_inputs, axis=0)
decoder_inputs = tf.unstack(decoder_inputs, axis=0)
results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        output_projection=None,
        feed_previous=False,
        dtype=None,
        scope=None
    )
logits = tf.stack(results, axis=0)
loss = tf.contrib.seq2seq.sequence_loss(logits, targets=targets, weights=weights)
pred = tf.argmax(logits, 2)
correct_pred = tf.equal(tf.cast(pred, tf.int64), tf.cast(targets, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()
with tf.Session() as sess:
    if os.path.exists(model_path):
        saver.restore(sess, model_path)
    else:
        sess.run(tf.global_variables_initializer())
    epoch = 0
    while epoch < 50:
        epoch = epoch + 1
        print("epoch:", epoch)
        for step in range(0, 1000):
            print("step:", step)
            train_x, train_y = loadQA()
            train_encoder_inputs = train_x[step * batch_size:step * batch_size + batch_size, :]
            train_targets = train_y[step * batch_size:step * batch_size + batch_size, :]
            train_decoder_inputs = np.zeros([batch_size, sequence_length])
            # results_value=sess.run(results,feed_dict={encoder_inputs:train_encoder_inputs,decoder_inputs:train_decoder_inputs})
            cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                             weights: train_weights, decoder_inputs: train_decoder_inputs})
            print(cost)
            accuracy_value = sess.run(accuracy, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                                           weights: train_weights,
                                                           decoder_inputs: train_decoder_inputs})
            print(accuracy_value)
            op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                               weights: train_weights, decoder_inputs: train_decoder_inputs})
            step = step + 1
        saver.save(sess, "./model/model.ckpt")
