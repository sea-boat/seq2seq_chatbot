import tensorflow as tf
import numpy as np

batch_size = 27
sequence_length = 10
hidden_size = 256
num_layers = 2
num_encoder_symbols = 1004  # 'UNK' and '<go>' and '<eos>' and '<pad>'
num_decoder_symbols = 1004
embedding_size = 256
learning_rate = 0.001
model_dir = './model'

encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)


def loadQA():
    train_x = np.load('./data/idx_q.npy', mmap_mode='r')
    train_y = np.load('./data/idx_a.npy', mmap_mode='r')
    train_target = np.load('./data/idx_o.npy', mmap_mode='r')
    return train_x, train_y, train_target


results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    tf.unstack(encoder_inputs, axis=1),
    tf.unstack(decoder_inputs, axis=1),
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    feed_previous=False
)
logits = tf.stack(results, axis=1)
print("sssss: ", logits)
loss = tf.contrib.seq2seq.sequence_loss(logits, targets=targets, weights=weights)
pred = tf.argmax(logits, axis=2)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()
train_weights = np.ones(shape=[batch_size, sequence_length], dtype=np.float32)
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    epoch = 0
    while epoch < 5000000:
        epoch = epoch + 1
        print("epoch:", epoch)
        for step in range(0, 1):
            print("step:", step)
            train_x, train_y, train_target = loadQA()
            train_encoder_inputs = train_x[step * batch_size:step * batch_size + batch_size, :]
            train_decoder_inputs = train_y[step * batch_size:step * batch_size + batch_size, :]
            train_targets = train_target[step * batch_size:step * batch_size + batch_size, :]
            op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                               weights: train_weights, decoder_inputs: train_decoder_inputs})
            cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
                                             weights: train_weights, decoder_inputs: train_decoder_inputs})
            print(cost)
            step = step + 1
        if epoch % 100 == 0:
            saver.save(sess, model_dir + '/model.ckpt', global_step=epoch + 1)
