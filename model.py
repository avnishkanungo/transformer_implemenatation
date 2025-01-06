import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Creating input embedding of the input sentences
class InputEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initializes the InputEmbedding module.

        Args:
            d_model (int): The dimensionality of the embedding vectors.
            vocab_size (int): The size of the vocabulary.

        Initializes an embedding layer and a dropout layer. The embedding layer maps vocabulary indices to
        embedding vectors of size `d_model`, and the dropout layer applies dropout regularization with a 
        probability of 0.1 to the embeddings.
        """

        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Computes the output of the input embedding module.

        Args:
            x (torch.Tensor): The input tensor containing vocabulary indices. (Batch, Seq_len)

        Returns:
            torch.Tensor: The output tensor with applied embedding and dropout, scaled by the square root of d_model. (Batch, Seq_len, d_model)
        """
        # if x.shape[1] != self.vocab_size:
        #     pad_size = self.vocab_size - x.shape[1]
        #     if pad_size > 0:
        #         x = F.pad(x, (0, pad_size))
        # print(x.shape)
        # print(self.vocab_size)
        x = self.embedding(x)
        # x = self.dropout(x)
        return x*math.sqrt(self.d_model)

# Creating positional encoding, which is added to the input embedding, to make the model understand the order of the words.
# So the final vector is a combination of the input embedding and the positional encoding. Embedded input + embedded information about position.
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            dropout (float, optional): The dropout rate. Default is 0.1.
            max_len (int, optional): The maximum length of the input sequences. Default is 5000.

        Initializes a dropout layer and a positional encoding tensor which is used to add positional information to the input embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #vector for an individual word's position
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator of the positional encoding formula
        pe[:, 0::2] = torch.sin(position * div_term) # pe formula for even indices
        pe[:, 1::2] = torch.cos(position * div_term) # pe formula for odd indices
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Computes the output of the positional encoding module.

        Args:
            x (torch.Tensor): The input tensor. (Batch, Seq_len, d_model)

        Returns:
            The output tensor. (Batch, Seq_len, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        Initializes the LayerNormalization module.

        Args:
            d_model (int): The dimensionality of the model.
            eps (float, optional): The epsilon value for numerical stability. Default is 1e-6.

        Initializes the learnable parameters a_2 and b_2, which are used to scale and shift the normalized vector.
        """
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(1))
        self.b_2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Computes the output of the layer normalization module.

        Args:
            x (torch.Tensor): The input tensor. (Batch, Seq_len, d_model)

        Returns:
            The output tensor. (Batch, Seq_len, d_model)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # a_2(is mulitipicative so initialize to 1) and b_2(is additive so initialized to 0) are learnable parameters, 
        # corresponding to gamma and beta, which can be used to amplify particular values or suppress others as needed by the model
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 

class FeedForwardBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes FeedForwardBlock module.

        Args:
            d_model (int): The dimensionality of the input and output vectors.
            d_ff (int): The dimensionality of the feedforward layer.
            dropout (float, optional): The dropout rate. Default is 0.1.
        """
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)# W1, B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2, B2

    def forward(self, x):
        #(Batch, Seq_len, d_model) --> RELU(Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        """
        Computes the output of the feedforward block.

        Args:
            x: The input tensor. (Batch, Seq_len, d_model)

        Returns:
            The output tensor. (Batch, Seq_len, d_model)
        """
        x = self.dropout(torch.relu(self.linear_1(x))) 
        x = self.linear_2(x)
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        """
        Initializes MultiHeadAttention module.

        Args:
            d_model (int): The dimensionality of the model.
            h (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Default is 0.1.

        Initializes the necessary weights and attention heads for the MultiHeadAttention module. The weights are
        initialized as Linear layers using PyTorch's nn.Linear, and the dropout is applied using PyTorch's nn.Dropout.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h" 
        self.d_model = d_model
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def Attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Computes attention between query and key, and applies it to value.

        Args:
            query (torch.Tensor): (Batch, h, Seq_len, d_k)
            key (torch.Tensor): (Batch, h, Seq_len, d_k)
            value (torch.Tensor): (Batch, h, Seq_len, d_v)
            mask (torch.ByteTensor, optional): (Batch, Seq_len, Seq_len)
            dropout (nn.Dropout, optional): The dropout layer to apply to the attention weights.

        Returns:
            output (torch.Tensor): (Batch, h, Seq_len, d_v)
            attention_weights (torch.Tensor): (Batch, h, Seq_len, Seq_len)
        """
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim=-1) # (Batch, h, Seq_len, Seq_len)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn



    def forward(self, q, k, v, mask=None):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)

        """
        Computes the MultiHeadAttention output.

        Args:
            q (torch.Tensor): The query tensor. (Batch, Seq_len, d_model)
            k (torch.Tensor): The key tensor. (Batch, Seq_len, d_model)
            v (torch.Tensor): The value tensor. (Batch, Seq_len, d_model)
            mask (torch.ByteTensor, optional): The attention mask. (Batch, Seq_len, Seq_len)

        Returns:
            output (torch.Tensor): The output tensor. (Batch, Seq_len, d_model)
            attention_weights (torch.Tensor): The attention weights. (Batch, h, Seq_len, Seq_len)
        """
        query, key, val = self.w_q(q), self.w_k(k), self.w_v(v)
         # (Batch, h, Seq_len, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        val = val.view(val.shape[0], val.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attn = MultiHeadAttention.Attention(query, key, val, mask, self.dropout)

        # (Batch, h, Seq_len, d_k) --> (Batch, Seq_len,h, d_k)
        # we use contiguous as transpose does not support in-place operations
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.d_model)
        self.last_layer_attention_weights = attn
        return self.w_o(x)


class SkipConnection(torch.nn.Module):
    def __init__(self, dropout: float = 0.1):
        """
        Initializes the SkipConnection module.

        Args:
            d_model (int): The dimensionality of the model.
            dropout (float, optional): The dropout rate. Default is 0.1.

        Initializes a dropout layer and a layer normalization instance to be used
        in the forward pass for implementing residual connections along with
        normalization.
        """
        super(SkipConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.LayerNormalization = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Applies a sublayer to the input and adds the result to the original input,
        followed by dropout. This implements a residual connection followed by
        layer normalization.

        Args:
            x: The input tensor.
            sublayer: A sublayer function to apply to the input tensor.

        Returns:
            A tensor with the same shape as `x`, after applying the sublayer,
            dropout, and adding the residual connection.
        """

        return x + self.dropout(sublayer(x))
    
class EncoderBlock(torch.nn.Module):
    def __init__(self,  self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout: float = 0.1 ):
    
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.SkipConnection = nn.ModuleList([SkipConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.SkipConnection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.SkipConnection[1](x, self.feed_forward_block)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderBlock(torch.nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout: float = 0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.SkipConnection = nn.ModuleList([SkipConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.SkipConnection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.SkipConnection[1](x, lambda x: self.cross_attention_block(x, enc_out, enc_out, src_mask))
        x = self.SkipConnection[2](x, self.feed_forward_block)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

class ProjectionLinearLayer(torch.nn.Module):
    def  __init__(self, dec_output_dim: int, vocab_size: int, dropout: float = 0.1):
        super(ProjectionLinearLayer, self).__init__()
        self.proj = nn.Linear(dec_output_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x = x.flatten(0, 1)
        # return self.dropout(self.proj(x))
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(torch.nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embedding:InputEmbedding, tgt_embedding:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection:ProjectionLinearLayer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.projection = projection

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.tgt_embedding(tgt)
        x = self.tgt_pos(x)
        x = self.decoder(x, enc_out, src_mask, tgt_mask)
        return x

    def project(self, x):
        return self.projection(x)

def buildTransformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8 , d_ff: int = 2048, N: int = 6, dropout: float = 0.1) -> Transformer:
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, dropout, src_seq_len)
    tgt_pos = PositionalEncoding(d_model, dropout, tgt_seq_len)
    # self_attention_block = MultiHeadAttention(d_model, h, dropout)
    # cross_attention_block = MultiHeadAttention(d_model, h, dropout)
    # feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    # encoder = Encoder([EncoderBlock(self_attention_block, feed_forward_block, dropout) for _ in range(N)])
    # decoder = Decoder([DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout) for _ in range(N)])
    encoder_blocks = [] 
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder)
    
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model, h, dropout)
        cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = ProjectionLinearLayer(d_model, tgt_vocab_size, dropout)

    transformer =  Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection)

    # Initialize the Transformer model parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer