import tensorflow as tf

import numpy as np
from word_id_test import Word_Id_Map

with tf.device('/cpu:0'):
    batch_size = 1
    sequence_length = 10
    num_encoder_symbols = 1004
    num_decoder_symbols = 1004
    embedding_size = 256
    hidden_size = 256
    num_layers = 2

    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
    weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        tf.unstack(encoder_inputs, axis=1),
        tf.unstack(decoder_inputs, axis=1),
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        feed_previous=True,
    )
    logits = tf.stack(results, axis=1)
    pred = tf.argmax(logits, axis=2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, module_file)
        map = Word_Id_Map()
        encoder_input = map.sentence2ids(['you', 'want', 'to', 'turn', 'twitter', 'followers', 'into', 'blog', 'readers'])

        encoder_input = encoder_input + [3 for i in range(0, 10 - len(encoder_input))]
        encoder_input = np.asarray([np.asarray(encoder_input)])
        decoder_input = np.zeros([1, 10])
        print('encoder_input : ', encoder_input)
        print('decoder_input : ', decoder_input)
        pred_value = sess.run(pred, feed_dict={encoder_inputs: encoder_input, decoder_inputs: decoder_input})
        print(pred_value)
        sentence = map.ids2sentence(pred_value[0])
        print(sentence)
