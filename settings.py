import os.path as osp

BASE_DIR = osp.dirname(osp.join(__file__))
DATA_DIR = osp.join(BASE_DIR, 'data')
EMBEDDINGS_DIR = osp.join(DATA_DIR, 'embeddings')
WORD2VEC_PATH = osp.join(EMBEDDINGS_DIR, 'Google-News-vectors-negative300.bin')
SAVED_MODELS = osp.join(BASE_DIR, 'saved_models')

VOCAB_SIZE = 10000
MAX_SEQ_LEN = 64

