import csv
import os
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer

from settings import VOCAB_SIZE, MAX_SEQ_LEN


def create_sentences_and_labels(file_path, train=False):
    file = open(file_path)
    read_tsv = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
    sentences = []
    classes = []
    tag_to_index = {}

    sentence = []
    clas = []
    for line in read_tsv:
        if line:
            word = line[0]
            tag = line[1]
            sentence.append(word)
            clas.append(tag)
            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index)
        else:
            sentences.append(sentence)
            classes.append(clas)
            sentence = []
            clas = []

    if train:
        index_to_tag = {(index, tag) for tag, index in tag_to_index.items()}
        return sentences, classes, tag_to_index, index_to_tag
    else:
        return sentences, classes


def create_vocab(sentences, vocab_limit):
    vocab = {}
    n_words = 0
    for sentence in sentences:
        for word in sentence:
            if word not in vocab and n_words <= vocab_limit:
                vocab[word] = n_words
                n_words += 1
    return vocab


def tags_to_indices(labels, tag_to_index):
    new_labels = []
    for label in labels:
        new_labels.append(tag_to_index[label])
    return new_labels


def transform_data(file_path, save_dir, train=False):
    if train:
        sentences, labels, tag_to_index, index_to_tag = create_sentences_and_labels(file_path, train)

        tokenizer = Tokenizer(num_words=VOCAB_SIZE, )
        tokenizer.fit_on_texts(sentences)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_dir, 'tag_to_index.pickle'), 'wb') as handle:
            pickle.dump(tag_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_dir, 'index_to_tag.pickle'), 'wb') as handle:
            pickle.dump(index_to_tag, handle, protocol=pickle.HIGHEST_PROTOCOL)

        vectors = tokenizer.texts_to_sequences(sentences)
        padded_vectors = pad_sequences(vectors, maxlen=MAX_SEQ_LEN, padding='post')

    else:
        sentences, labels = create_sentences_and_labels(file_path, train)

        with open(os.path.join(save_dir, 'tokenizer.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(os.path.join(save_dir, 'tag_to_index.pickle'), 'rb') as handle:
            tag_to_index = pickle.load(handle)
        with open(os.path.join(save_dir, 'index_to_tag.pickle'), 'rb') as handle:
            index_to_tag = pickle.load(handle)

        vectors = tokenizer.texts_to_sequences(sentences)
        padded_vectors = pad_sequences(vectors, maxlen=MAX_SEQ_LEN, padding='post')

    # new_labels = [pad_sequences(maxlen=MAX_SEQ_LEN,
    #                             padding="post",
    #                             value=tag_to_index["O"],
    #                             sequences=to_categorical(tags_to_indices(sentence_labels, tag_to_index),
    #                                                      num_classes=len(tag_to_index)))
    #               for sentence_labels in labels]
    new_labels = [to_categorical(tags_to_indices(sentence_labels, tag_to_index), num_classes=len(tag_to_index))
                  for sentence_labels in labels]

    new_labels = pad_sequences(maxlen=MAX_SEQ_LEN, sequences=new_labels, padding="post", value=tag_to_index["O"])

    return padded_vectors, new_labels, tag_to_index, index_to_tag
