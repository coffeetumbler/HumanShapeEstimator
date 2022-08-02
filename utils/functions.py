import numpy as np
import copy

import torch
import torch.nn as nn



# Make clones of a layer.
def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# # Find maximum length of sequence and make masking matrix from the given information of sequence length.
# def get_seq_info(seq_len_info, using_CLS, device=None):
#     """
#     <input>
#     seq_len_info : (n_batch), seq_len_info[i] = length of i-th sequence, numpy array type
#     using_CLS : dtype=bool, CLS option
#     device : cpu or cuda, device for mask
    
#     <output>
#     max_seq_len : maximum length of the sequences
#     mask : (n_batch, 1, 1, max_seq_len), True values for padding values, CLS column is added if CLS option is on.
#     """
#     max_seq_len = np.max(seq_len_info)
#     if using_CLS:
#         mask = torch.ones(len(seq_len_info), max_seq_len+1, dtype=torch.bool, device=device)
#         for i, seq_len in enumerate(seq_len_info):
#             mask[i, 1:seq_len+1] = False  # :seq_len+1 -> 1:seq_len+1
#         return max_seq_len, mask.view(-1, 1, 1, max_seq_len+1).contiguous()
#     else:
#         mask = torch.ones(len(seq_len_info), max_seq_len, dtype=torch.bool, device=device)
#         for i, seq_len in enumerate(seq_len_info):
#             mask[i, :seq_len] = False
#         return max_seq_len, mask.view(-1, 1, 1, max_seq_len).contiguous()


# # Add extra padding as the first element of data and insert CLS token.
# def insert_CLS(x):
#     out = nn.functional.pad(x, (1, 0, 1, 0), value=0)  # Add zero padding to both first row and column.
#     # Insert the first standard basis vector as CLS token.
#     mask = torch.zeros_like(out, device=x.device)
#     if mask.dim() == 2:
#         mask[0, 0] = 1
#     elif mask.dim() == 3:
#         mask[:, 0, 0] = 1
#     return out + mask


# # Make a row operation matrix which transforms data matrix.
# def get_row_op(n_batch, seq_len_info, n_token, max_seq_len, using_CLS, device=None):
#     """
#     <input>
#     n_batch : number of sequences
#     seq_len_info : (n_batch), seq_len_info[i] = length of i-th sequence, numpy array type
#     n_token : total number of tokens(images)
#     max_seq_len : maximum length of the sequences
#     using_CLS : dtype=bool, CLS option
#     device : cpu or cuda, device for row operation
#     """
#     if using_CLS:
#         max_seq_len += 1
#         k = 1
#         l = 1
#     else:
#         k = 0
#         l = 0
    
#     row_operation = torch.zeros(n_batch * max_seq_len, n_token, device=device)
#     if using_CLS:
#         row_operation[::max_seq_len, 0] = 1

#     for seq_len in seq_len_info:
#         row_operation[k:k+seq_len, l:l+seq_len] = torch.eye(seq_len, device=device)
#         k += max_seq_len
#         l += seq_len
        
#     return row_operation