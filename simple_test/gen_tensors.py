from simple_test_lib import get_config, save_data
import numpy as np


def main():
    config = get_config()

    msl = config['msl']
    embedding_dim = config['embedding_dim']
    batch_size = config['batch_size']
    num_heads = config['num_heads']
    ff_ex_factor = config['ff_ex_factor']

    d = embedding_dim//num_heads

    # Generate needed tensors
    # MultiHeadAttention
    Wq = np.random.random((embedding_dim, num_heads*d)).astype(np.float32)
    Bq = np.random.random((num_heads*d,)).astype(np.float32)
    Wk = np.random.random((embedding_dim, num_heads*d)).astype(np.float32)
    Bk = np.random.random((num_heads*d,)).astype(np.float32)
    Wv = np.random.random((embedding_dim, num_heads*d)).astype(np.float32)
    Bv = np.random.random((num_heads*d,)).astype(np.float32)
    Wo = np.random.random((num_heads*d, embedding_dim)).astype(np.float32)
    Bo = np.random.random((embedding_dim,)).astype(np.float32)

    # Expansion Layer
    W1 = np.random.random((embedding_dim, ff_ex_factor*embedding_dim)).astype(np.float32)
    B1 = np.random.random((ff_ex_factor*embedding_dim,)).astype(np.float32)
    W2 = np.random.random((ff_ex_factor*embedding_dim, embedding_dim)).astype(np.float32)
    B2 = np.random.random((embedding_dim,)).astype(np.float32)

    # test input data
    test_data = np.random.random((batch_size, msl, embedding_dim)).astype(np.float32)

    tensors = {
        'Wq': Wq,
        'Bq': Bq,
        'Wk': Wk,
        'Bk': Bk,
        'Wv': Wv,
        'Bv': Bv,
        'Wo': Wo,
        'Bo': Bo,
        'W1': W1,
        'B1': B1,
        'W2': W2,
        'B2': B2,
        'data': test_data,}

    save_data(tensors, config=config)


if __name__ == "__main__":
    main()
