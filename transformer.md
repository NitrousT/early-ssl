# Vision Transformer Details


## Vision Transformer Class:
The input parameters of the class are:
* img_size (list of int): The size of the input image (default: [224]).
* patch_size (int): The size of the patches used for the patch embeddings (default: 16).
* in_chans (int): The number of input channels (default: 3).
* num_classes (int): The number of output classes (default: 0).
* embed_dim (int): The dimension of the patch embeddings (default: 768).
* depth (int): The number of transformer blocks in the model (default: 12).
* num_heads (int): The number of attention heads in the multi-head attention layer (default: 12).
* mlp_ratio (float): The ratio of the hidden dimension to the embedding dimension in the feedforward network (default: 4.0).
* qkv_bias (bool): Whether to include bias terms in the linear projection layers for query, key, and value (default: False).
* qk_scale (float): The scale factor for the dot product attention (default: None).
* drop_rate (float): The dropout rate for the patches and the classification token (default: 0.0).
* attn_drop_rate (float): The dropout rate for the attention scores (default: 0.0).
* drop_path_rate (float): The probability of dropping out a transformer block (default: 0.0).
* norm_layer (nn.Module): The normalization layer used in the model (default: nn.LayerNorm).

### _init_weights :
he _init_weights method initializes the weights and biases of linear and layer normalization layers with truncated normal distribution and constant values, respectively. This method is called during the initialization of the VisionTransformer module to initialize all its sub-modules.

Specifically, for a linear layer, the method sets the weights with values sampled from a truncated normal distribution with standard deviation 0.02 and sets the biases to zero. For a layer normalization layer, the method sets the biases to zero and sets the weights to 1.0.

### Interpolate_pos_encoding :
The interpolate_pos_encoding method takes as input the output tensor x and the width w and height h of the input image. It first checks if the number of patches in x and the size of the positional embedding tensor self.pos_embed match. If they do, it returns self.pos_embed as is. Otherwise, it extracts the class position embedding from self.pos_embed and the patch position embeddings from the remaining entries, and interpolates the patch position embeddings to match the number and size of patches in x.

The patch position embeddings are first reshaped to a square grid, with one row and column per side, and then transposed to be in the format expected by the nn.functional.interpolate function. The scale factor for the interpolation is computed as the ratio of the input image size to the size of the patch grid. The interpolation is performed using bicubic interpolation, which is a type of spatial interpolation commonly used in image processing. Finally, the interpolated patch position embeddings are concatenated with the class position embedding and returned.

### prepare_tokens :
The prepare_tokens method takes an input tensor x of shape [batch_size, num_channels, image_height, image_width] as argument, performs a patch embedding operation on the input, adds a [CLS] token to the resulting embeddings, adds positional encoding to each token, and returns the result after applying dropout.

### get_last_selfattention :
The get_last_selfattention method takes an input tensor x and applies the DINO model to it, returning the output tensor of the last block and the self-attention scores of that block. The attention scores are returned as a tuple with the output tensor. This method can be used to visualize the attention scores and analyze the model's behavior.

