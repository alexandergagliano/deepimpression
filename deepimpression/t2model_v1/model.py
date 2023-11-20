import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mk_ic import install
install()


class ConvEmbedding_AG(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(ConvEmbedding_AG, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,  # This might need to be adjusted depending on the input size
            out_channels=num_filters,
            kernel_size=1,
        )
        self.activation = F.gelu

    def forward(self, inputs):
        # Assuming inputs is of shape (batch, channels, sequence_length)
        return self.activation(self.conv1d(inputs))


class PositionalEncoding(nn.Module):
    def __init__(self, max_steps, max_dims):
        super(PositionalEncoding, self).__init__()
        self.max_steps = max_steps
        self.max_dims = max_dims

        if max_dims % 2 == 1:
            max_dims += 1  # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_dims, max_steps))
        pos_emb[0, ::2, :] = np.sin(p / 10000 ** (2 * i / max_dims))
        pos_emb[0, 1::2, :] = np.cos(p / 10000 ** (2 * i / max_dims))
        self.register_buffer('positional_embedding', torch.tensor(pos_emb, dtype=torch.float))

    def forward(self, inputs):
        # inputs is of shape (batch, sequence_length, model_dim)
        return inputs + self.positional_embedding[:, :inputs.size(1), :]

class TransformerBlock_AG(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock_AG, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs, key_padding_mask=None, training=True):
        # Assuming inputs is of shape (sequence_length, batch, model_dim)
        attn_output, _ = self.att(inputs, inputs, inputs, key_padding_mask=key_padding_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


def compute_mask(data, mask_value=0.):
    # Assuming data is of shape (batch, sequence_length, model_dim)
    #return torch.all(torch.eq(data, mask_value), -1)
    # Assuming data is of shape (batch, model_dim, sequence_length)
    return torch.all(torch.eq(data, mask_value), -2)


class T2Model_AG(nn.Module):
    # TODO: Update docstrings
    """Time-Transformer with Multi-headed.
    embed_dim --> Embedding size for each token
    num_heads --> Number of attention heads
    ff_dim    --> Hidden layer size in feed forward network inside transformer
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        ff_dim,
        num_filters,
        num_classes,
        num_layers,
        droprate,
        num_aux_feats=0,
        add_aux_feats_to="M",
        **kwargs,
    ):
        super(T2Model_AG, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.droprate = droprate
        self.num_aux_feats = num_aux_feats
        self.add_aux_feats_to = add_aux_feats_to

        self.num_classes = num_classes
        self.sequence_length = input_dim[2] + self.num_aux_feats if self.add_aux_feats_to == "L" else input_dim[2]
        ic(self.sequence_length)

        self.embedding = ConvEmbedding_AG(in_channels=input_dim[1], num_filters=self.num_filters)  # A custom class to be defined
        self.pos_encoding = PositionalEncoding(max_steps=self.sequence_length, max_dims=self.embed_dim)  # A custom class to be defined

        self.encoder = nn.ModuleList([
            TransformerBlock_AG(self.embed_dim, self.num_heads, self.ff_dim) for _ in range(num_layers)  # A custom class to be defined
        ])

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(self.droprate)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x, training=None):
        # Implement the forward pass
        if isinstance(x, torch.Tensor):
            # Compute key_padding_mask
            key_padding_mask = compute_mask(x)

            # Assume x is the input tensor
            x = self.embedding(x)
            x = self.pos_encoding(x)
            # conv layer expects channels first, but attention expects channels last
            x = x.swapaxes(1, 2)

            for layer in self.encoder:
                x = layer(x, key_padding_mask=key_padding_mask)

            # Swap axes back

            x = x.mean(dim=1) if self.pooling is None else self.pooling(x.transpose(1, 2)).squeeze(-1)
            x = self.dropout1(x)
            x = self.classifier(x)

        else:
            # Handle the case where x is a list or a dict
            # This part would need to be implemented based on how the inputs are expected to be structured in PyTorch
            pass

        return x

