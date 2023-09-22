# Deep Impression
A transformer-based transient classifier, trained using a slightly tweaked version of [`astronet`](https://github.com/tallamjr/astronet). This model has been adapted to use tensorflow's new MultiHeadAttention layer (which allows for padding masks). This makes it able to use partial-phase photometry and physical timescales of transients for classification!
