import torch
import numpy as np
from simple_test_lib import get_config, load_data
from mk_ic import install
install()


def compute_mask(data, mask_value=0.):
    #boolean_mask = tf.reduce_any(
    #    tf.not_equal(inputs, self.mask_value), axis=-1, keepdims=True
    #)
    neq = torch.eq(data, mask_value)
    num_nonzero = torch.sum(neq)
    #bool_mask = torch.any(neq, -1, keepdim=True)
    bool_mask = torch.all(neq, -1)
    #outputs = inputs * tf.cast(boolean_mask, inputs.dtype)
    ## Compute the mask and outputs simultaneously.
    #outputs._keras_mask = tf.squeeze(boolean_mask, axis=-1)
    return bool_mask


class TestClass(torch.nn.Module):
    def __init__(self, embed_dim=32, num_heads=1, ff_ex_factor=2, **kwargs):
        super().__init__()
        self.layer = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, **kwargs)
        self.ex1 = torch.nn.Linear(embed_dim, ff_ex_factor*embed_dim)
        self.ex2 = torch.nn.Linear(ff_ex_factor*embed_dim, embed_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs, **kwargs):
        # Build mask

        o = self.layer(inputs, inputs, inputs, **kwargs)
        #o = self.ex2(self.relu(self.ex1(inputs)))
        return o
        #return self.layer([inputs, inputs, inputs])


def main():
    config = get_config()
    msl = config['msl']
    embedding_dim = config['embedding_dim']
    batch_size = config['batch_size']
    num_heads = config['num_heads']
    ff_ex_factor = config['ff_ex_factor']

    data_dict = load_data(config=config)

    # Load data into tensor
    test_data = data_dict['data']
    test_data[:,2,:] = 0.
    test_data = torch.tensor(test_data)

    ic("Data Dict")
    for key, val in data_dict.items():
        ic(key, val.shape)

    data_mask = compute_mask(test_data)

    ic(data_mask)

    mdl = TestClass(embed_dim=embedding_dim, num_heads=num_heads, ff_ex_factor=ff_ex_factor)

    # Build up in_proj weights
    in_proj_weight = np.concatenate(
        (data_dict['Wq'].transpose(),
         data_dict['Wk'].transpose(),
         data_dict['Wv'].transpose()), axis=0)
    in_proj_weight = torch.from_numpy(in_proj_weight)
    in_proj_bias = np.concatenate(
        (data_dict['Bq'],
         data_dict['Bk'],
         data_dict['Bv']), axis=0)
    in_proj_bias = torch.from_numpy(in_proj_bias)
    Wo = data_dict['Wo']
    ic(Wo.shape)
    out_proj_weight = torch.from_numpy(data_dict['Wo'].transpose())
    #out_proj_weight = torch.from_numpy(data_dict['Wo'])
    out_proj_bias = torch.from_numpy(data_dict['Bo'])

    ic(data_dict['W1'].shape)
    W1 = torch.from_numpy(data_dict['W1'].transpose())
    B1 = torch.from_numpy(data_dict['B1'])
    W2 = torch.from_numpy(data_dict['W2'].transpose())
    B2 = torch.from_numpy(data_dict['B2'])

    # Copy weights into layers
    with torch.no_grad():
        mdl.layer.in_proj_weight.copy_(in_proj_weight)
        mdl.layer.in_proj_bias.copy_(in_proj_bias)
        mdl.layer.out_proj.weight.copy_(out_proj_weight)
        mdl.layer.out_proj.bias.copy_(out_proj_bias)
        mdl.ex1.weight.copy_(W1)
        mdl.ex1.bias.copy_(B1)
        mdl.ex2.weight.copy_(W2)
        mdl.ex2.bias.copy_(B2)

    ic(test_data)
    results = mdl(test_data, need_weights=True, key_padding_mask=data_mask)
    ic(results[0], results[0].shape, results[1], results[1].shape)
    #ic(results, results.shape)
    #for name, parameter in mdl.named_parameters():
    #    ic(name, parameter.shape, parameter)

    #raise RuntimeError("Early Stop")

    #mdl = TestClass(input_shape=test_shape)

    #ic(test_data, data_mask)
    #result = mdl(test_data, need_weights=True, key_padding_mask=data_mask)
    #result = mdl(test_data)

    #ic(result)

    #mdl.summary(expand_nested=True)

    #total_pars = 0
    #for name, parameter in mdl.named_parameters():
    #    total_pars += np.prod(parameter.shape)
    #    ic(name, parameter, parameter.shape)
    #ic(total_pars)


if __name__ == "__main__":
    main()
