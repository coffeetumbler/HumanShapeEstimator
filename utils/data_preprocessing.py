"""
Parts of the code are taken from https://github.com/microsoft/MeshTransformer
"""

import pickle, torch
import numpy as np

model_path = 'data/smpl/'
models = {'male' : 'basicModel_m_lbs_10_207_0_v1.0.0.pkl',
          'female' : 'basicModel_f_lbs_10_207_0_v1.0.0.pkl',
          'neutral' : 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'}

for key, value in models.items():
    with open(model_path + value, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        smpl_model = u.load()
        
    J_regressor = smpl_model['J_regressor'].tocoo()

    row = J_regressor.row
    col = J_regressor.col
    data = J_regressor.data
    i = torch.LongTensor([row, col])
    v = torch.FloatTensor(data)
    J_regressor_shape = [24, 6890]

    item = {}
    item['J_regressor'] = torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense()
    
    item['weights'] = torch.FloatTensor(smpl_model['weights'])
    item['posedirs'] = torch.FloatTensor(smpl_model['posedirs'])
    item['v_template'] = torch.FloatTensor(smpl_model['v_template'])
    item['shapedirs'] = torch.FloatTensor(np.array(smpl_model['shapedirs']))
    kintree_table = torch.from_numpy(smpl_model['kintree_table'].astype(np.int64))
    id_to_col = {kintree_table[1, i].item(): i for i in range(kintree_table.shape[1])}
    item['parent'] = torch.LongTensor([id_to_col[kintree_table[0, it].item()] for it in range(1, kintree_table.shape[1])])

    faces = smpl_model['f'].astype(np.int64)
    
    torch.save(item, 'data/J_regressor_'+key+'.pt')
    if key == 'neutral':
        np.save('data/smpl_faces_neutral', faces)