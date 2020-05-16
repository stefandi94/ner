import datetime
import os.path as osp

import numpy as np

from models.bi_lstm import BiLSTM
from preprocessing import transform_data
from settings import DATA_DIR, SAVED_MODELS, MAX_SEQ_LEN

EPOCHS = 5
BATCH_SIZE = 128
N_NEURONS = 128

if __name__ == '__main__':
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = osp.join(SAVED_MODELS, time)

    train_path = osp.join(DATA_DIR, 'train.tsv')
    val_path = osp.join(DATA_DIR, 'dev.tsv')
    X_train, y_train, tag_to_index, index_to_tag = transform_data(train_path,
                                                                  save_dir=save_dir,
                                                                  train=True)

    X_valid, y_valid, _, _ = transform_data(val_path, save_dir=save_dir, train=False)

    model = BiLSTM(MAX_SEQ_LEN, n_neurons=N_NEURONS, n_classes=len(tag_to_index))
    model.fit(X_train, np.array(y_train),
              X_valid, y_valid,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              save_dir=save_dir)

    test_path = osp.join(DATA_DIR, 'test.tsv')
    X_test, y_test, _, _ = transform_data(test_path, save_dir=save_dir, train=False)

