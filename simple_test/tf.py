import tensorflow as tf
import numpy as np
from simple_test_lib import get_config, load_data
from mk_ic import install
install()


class TestClass(tf.keras.Model):
    def __init__(self, embedding_dim=4, num_heads=2, ff_ex_factor=2, **kwargs):
        super().__init__()
        self.masking_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads, value_dim=embedding_dim//num_heads, **kwargs)
        self.exp1 = tf.keras.layers.Dense(embedding_dim*ff_ex_factor, activation='relu')
        self.exp2 = tf.keras.layers.Dense(embedding_dim, activation='linear')
        #self.layer = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=32, **kwargs)
        #self.layer = tf.keras.layers.Attention(key_dim=32, **kwargs)
        #self.layer = tf.keras.layers.Attention(**kwargs)

    def call(self, inputs, training=False, **kwargs):
        var1 = self.masking_layer(inputs)
        ic(var1._keras_mask)
        num_unmasked = tf.math.reduce_sum(tf.cast(var1._keras_mask, dtype='float32')).numpy()
        total_size = tf.math.reduce_prod(var1._keras_mask.shape).numpy()
        ic(var1, num_unmasked, total_size)
        var2 = self.layer(var1, var1, var1, **kwargs)
        #num_unmasked = tf.math.reduce_sum(tf.cast(var2._keras_mask, dtype='float32')).numpy()
        #total_size = tf.math.reduce_prod(var2._keras_mask.shape).numpy()
        #ic(var2, num_unmasked, total_size)
        #is_equal = tf.math.reduce_all(tf.math.equal(var1._keras_mask, var2._keras_mask)).numpy()
        #ic(is_equal)
        #var3 = self.exp1(var2)
        #var4 = self.exp2(var3)
        #num_unmasked = tf.math.reduce_sum(tf.cast(var4._keras_mask, dtype='float32')).numpy()
        #total_size = tf.math.reduce_prod(var4._keras_mask.shape).numpy()
        #is_equal = tf.math.reduce_all(tf.math.equal(var1._keras_mask, var4._keras_mask)).numpy()
        #ic(var4, num_unmasked, total_size, is_equal)
        #return var4
        #return self.layer([inputs, inputs, inputs])
        return var2


def main():
    config = get_config()
    msl = config['msl']
    embedding_dim = config['embedding_dim']
    test_shape = (msl, embedding_dim)
    batch_size = config['batch_size']
    num_heads = config['num_heads']
    ff_ex_factor = config['ff_ex_factor']

    data_dict = load_data(config=config)

    # Load data into tensor
    test_data = data_dict['data']
    test_data[:,2,:] = 0.
    test_data = tf.convert_to_tensor(test_data)

    ic("Data Dict")
    for key, val in data_dict.items():
        ic(key, val)

    mdl = TestClass(
        input_shape=test_shape,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_ex_factor=ff_ex_factor)
    # Evaluate to initialize graph
    temp_result = mdl(test_data, return_attention_scores=True)
    att_weights = tf.reduce_mean(temp_result[1], axis=1)

    ic("INITIAL RESULT", temp_result[0], att_weights)

    head_d = embedding_dim // num_heads

    # Query transform
    Wq = data_dict['Wq']
    Wq = Wq.reshape((embedding_dim, head_d, head_d))
    Bq = data_dict['Bq']
    Bq = Bq.reshape((head_d, head_d))

    # Key transform
    Wk = data_dict['Wk']
    Wk = Wk.reshape((embedding_dim, head_d, head_d))
    Bk = data_dict['Bk']
    Bk = Bk.reshape((head_d, head_d))

    # Value transform
    Wv = data_dict['Wv']
    Wv = Wv.reshape((embedding_dim, head_d, head_d))
    Bv = data_dict['Bv']
    Bv = Bv.reshape((head_d, head_d))

    # Output transform
    Wo = data_dict['Wo']
    Wo = Wo.reshape((head_d, head_d, embedding_dim))
    Bo = data_dict['Bo']

    # exp1 layer
    W1 = data_dict['W1']
    B1 = data_dict['B1']
    ic(W1, W1.shape, B1, B1.shape)

    # exp2 layer
    W2 = data_dict["W2"]
    B2 = data_dict["B2"]

    mdl.layer.set_weights([Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo])
    #mdl.exp1.set_weights([W1, B1])
    #mdl.exp2.set_weights([W2, B2])

    # View initialized weights
    ic(mdl.trainable_variables)
    ic(test_data)

    result = mdl(test_data, return_attention_scores=True)
    att_weights = tf.reduce_mean(result[1], axis=1)
    #result = mdl(test_data)

    ic("FINAL RESULT", result[0], att_weights)


if __name__ == "__main__":
    main()
