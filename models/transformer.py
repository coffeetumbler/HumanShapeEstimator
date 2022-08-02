import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# Make clones of a layer.
def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



# Main transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer, return_position_vector=False):
        super(TransformerEncoder, self).__init__()
        self.return_position_vector = return_position_vector
        self.encoder_layers = clone_layer(encoder_layer, n_layer)
            
        self.positional_encoding_layer = positional_encoding_layer if positional_encoding_layer is not None else False
        
    def forward(self, x, n_batch, masking_matrix, z=None):
        """
        <input>
        x : (n_token, d_embed), n_token = sum of length of all sequences
        n_batch : number of sequences in a batch
        masking_matrix : (n_batch, 1, 1, seq_len), True values for padding values
        z : input for positional encoding layer, None for no positional encoding or the same input with x
        """
        if self.positional_encoding_layer:
            out, position_vector = self.positional_encoding_layer(x, z)
        else:
            out = x
            
        out = out.view(n_batch, x.shape[0] // n_batch, -1).contiguous()

        for layer in self.encoder_layers:
            out = layer(out, n_batch, masking_matrix)

        if self.return_position_vector:
            return out, position_vector  # Return position vector with output.
        else:
            return out
    
    
# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        
    def forward(self, x, n_batch, masking_matrix):
        out1 = self.attention_layer(x, n_batch, masking_matrix)
        out1 = self.dropout_layer(out1) + x
        out1 = self.norm_layers[0](out1)
        
        out2 = self.feed_forward_layer(out1)
        out2 = self.dropout_layer(out2) + out1
        return self.norm_layers[1](out2)
    
    
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, d_model, n_head):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % n_head == 0  # Ckeck if d_model is divisible by n_head.
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.scale = 1 / np.sqrt(self.d_k)
        
        self.word_fc_layers = clone_layer(nn.Linear(d_embed, d_model), 3)
        self.output_fc_layer = nn.Linear(d_model, d_embed)

    def forward(self, x, n_batch, masking_matrix):
        """
        <input>
        x : (n_batch, seq_len, d_embed), seq_len = length of a sequence
        n_batch : number of sequences in a batch
        masking_matrix : (n_batch, 1, 1, seq_len), True values for padding values
        """        
        # Apply linear layers.
        query = self.word_fc_layers[0](x)
        key = self.word_fc_layers[1](x)
        value = self.word_fc_layers[2](x)
        
        # Split heads.
        query_out = query.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        key_out = key.view(n_batch, -1, self.n_head, self.d_k).contiguous().permute(0, 2, 3, 1)
        value_out = value.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute attention and concatenate matrices.
        scores = torch.matmul(query_out, key_out) * self.scale
        if masking_matrix != None:
            scores = scores + masking_matrix * (-1e9) # Add very small negative number to padding columns.
        probs = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(probs, value_out)
        
        # Convert 4d tensor to proper 3d output tensor.
        attention_out = attention_out.transpose(1, 2).contiguous().view(n_batch, -1, self.d_model)
            
        return self.output_fc_layer(attention_out)

    
    
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.dropout_layer(self.activation_layer(out))
        return self.second_fc_layer(out)
    
    
    
# Vector concatenating encoding
class VectorConcatenatingEncoding(nn.Module):
    def __init__(self, position_vector_layer, encoding_trainable, dropout=0.1):
        super(VectorConcatenatingEncoding, self).__init__()
        self.position_vector_layer = position_vector_layer
        self.encoding_trainable = encoding_trainable
        if not encoding_trainable:
            for params in self.position_vector_layer.parameters():
                params.requires_grad = False
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def forward(self, x, z=None):
        if self.encoding_trainable:
            position_vector = self.position_vector_layer(z)
        else:
            with torch.no_grad():
                position_vector = self.position_vector_layer(z)
        out = torch.cat([x, position_vector], dim=-1)        
        return self.dropout_layer(out), position_vector
    
    

# Get a transformer encoder with its parameters.
def get_transformer_encoder(d_embed,
                            positional_encoding='Concatenating',
                            position_vector_layer=None,
                            encoding_trainable=True,
                            return_position_vector=False,
                            n_layer=6,
                            d_model=2048,
                            n_head=8,
                            d_ff=2048,
                            max_seq_len=24,
                            dropout=0.1):
    
    if positional_encoding == 'Concatenating' or positional_encoding =='concatenating' or positional_encoding == 'cat':
        positional_encoding_layer = VectorConcatenatingEncoding(position_vector_layer, encoding_trainable, dropout)
        d_embed += position_vector_layer.position_vector_len
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None
    
    attention_layer = MultiHeadAttentionLayer(d_embed, d_model, n_head)
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)
    
    return TransformerEncoder(positional_encoding_layer, encoder_layer, n_layer, return_position_vector)
