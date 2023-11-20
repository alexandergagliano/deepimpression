import argparse
import pickle


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msl", help="max sequence length", default=5, type=int)
    parser.add_argument("--embedding_dim", help="Embedding dim", default=4, type=int)
    parser.add_argument("--batch_size", help="Batch size", default=4, type=int)
    parser.add_argument("--num_heads", help="Number of heads", default=2, type=int)
    parser.add_argument("--ff_ex_factor", help="expansion factor for expansion block", default=4)

    args = parser.parse_args()
    return {
        'msl': args.msl,
        'embedding_dim': args.embedding_dim,
        'batch_size': args.batch_size,
        'num_heads': args.num_heads,
        'ff_ex_factor': args.ff_ex_factor, }


def get_save_name(base="data", config={}):
    return f"{base}_msl{config['msl']}_ed{config['embedding_dim']}_bs{config['msl']}_nh{config['num_heads']}_ff{config['ff_ex_factor']}.pkl"


def save_data(data, config={}, base="data"):
    with open(get_save_name(base=base, config=config), 'wb') as f:
        f.write(pickle.dumps(data))


def load_data(config={}, base="data"):
    with open(get_save_name(base=base, config=config), 'rb') as f:
        data = pickle.loads(f.read())
    return data
