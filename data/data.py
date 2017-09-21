import nltk
import itertools
import numpy as np
import pickle

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'conv.txt'

limit = {
    'maxq': 10,
    'minq': 0,
    'maxa': 8,
    'mina': 3
}

UNK = 'unk'
GO = '<go>'
EOS = '<eos>'
PAD = '<pad>'
VOCAB_SIZE = 1000


def read_lines(filename):
    return open(filename, encoding='UTF-8').read().split('\n')


def split_line(line):
    return line.split('.')


def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = [GO] + [EOS] + [UNK] + [PAD] + [x[0] for x in vocab]
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist


def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i + 1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i + 1])
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)
    # +2 dues to '<go>' and '<eos>'
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    idx_o = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'], 1)
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 2)
        o_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 3)
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_o[i] = np.array(o_indices)

    return idx_q, idx_a, idx_o


def pad_seq(seq, lookup, maxlen, flag):
    if flag == 1:
        indices = []
    elif flag == 2:
        indices = [lookup[GO]]
    elif flag == 3:
        indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    if flag == 1:
        return indices + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 2:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq))
    elif flag == 3:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (maxlen - len(seq) + 1)


def process_data():
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)
    lines = [line.lower() for line in lines]
    print('\n>> Filter lines')
    lines = [filter_line(line, EN_WHITELIST) for line in lines]
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines)
    print('\nq : {0} '.format(qlines[:]))
    print('\na : {0}'.format(alines[:]))
    print('\n>> Segment lines into words')
    qtokenized = [wordlist.split(' ') for wordlist in qlines]
    atokenized = [wordlist.split(' ') for wordlist in alines]
    print('\n:: Sample from segmented list of words')
    print('\nq : {0}'.format(qtokenized[:]))
    print('\na : {0}'.format(atokenized[:]))

    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)
    print(idx2w)
    with open('idx2w.pkl', 'wb') as f:
        pickle.dump(idx2w, f)
    with open('w2idx.pkl', 'wb') as f:
        pickle.dump(w2idx, f)
    idx_q, idx_a, idx_o = zero_pad(qtokenized, atokenized, w2idx)

    print('\nq : {0} ; a : {1} : o : {2}'.format(idx_q[0], idx_a[0], idx_o[0]))
    print('\nq : {0} ; a : {1} : o : {2}'.format(idx_q[1], idx_a[1], idx_o[1]))

    print('\n >> Save numpy arrays to disk')
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    np.save('idx_o.npy', idx_o)

    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_data(PATH=''):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


if __name__ == '__main__':
    process_data()
