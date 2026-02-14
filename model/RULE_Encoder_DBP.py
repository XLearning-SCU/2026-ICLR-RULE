from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import math

from .Tool_model import GAT, GCN
from .gcn_layer import NR_GraphAttention, NR_GraphAttentionCross

class RULEusion(nn.Module):
    '''
    RULEusion class for multi-modal feature fusion
    '''
    
    def __init__(self):
        
        """
        Args:
            args: Model arguments containing configurations.
            ent_num: Number of entities.
            modal_num: Number of modalities.
            with_weight: Whether to apply weights in normalization (default is 1).
        """
        super().__init__()

    def forward(self, embs_dict, mask):
     
        embs = [emb for key,emb in embs_dict.items() if emb is not None]
        

        embs = [
                F.normalize(emb, dim=1) * mask[key].unsqueeze(1).repeat(1, emb.shape[1]) 
                for key, emb in embs_dict.items() if emb is not None
            ]

        # embs = [F.normalize(emb) for key, emb in embs_dict.items() if emb is not None]

        joint_emb = torch.cat(embs, dim=1)
       
        return joint_emb

class RULE_Encoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units
    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 attr_input_dim=1000,
                 triple_size=10000,
                 node_size=10000,
                 rel_size=300,
                 ):
        super(RULE_Encoder, self).__init__()

        self.args = args
        self.ENT_NUM = ent_num
        self.dropout = nn.Dropout(self.args.dropout)
        self.node_size=node_size
        self.rel_size=rel_size
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])
        self.depth = 1

        output_dim = self.input_dim * int(self.depth + 1)

        # Initialize embedding layers and FC layers
        self.ent_embedding = nn.Embedding(self.ENT_NUM, self.input_dim)
        self.rel_embedding = nn.Embedding(rel_size, self.input_dim)
        self.img_fc = nn.Linear(img_feature_dim, output_dim)
        self.att_fc = nn.Linear(attr_input_dim, output_dim)

        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
        torch.nn.init.xavier_uniform_(self.img_fc.weight)
        torch.nn.init.xavier_uniform_(self.att_fc.weight)

        self.e_encoder_cross = NR_GraphAttentionCross(node_size=self.node_size,
                                           rel_size=rel_size,
                                           triple_size=triple_size,
                                           node_dim=self.input_dim,
                                           depth=self.depth,
                                           use_bias=True
                                           )

        # self.e_encoder_img_cross = NR_GraphAttentionCross(node_size=self.node_size,
        #                                    rel_size=rel_size,
        #                                    triple_size=triple_size,
        #                                    node_dim=self.input_dim,
        #                                    depth=0,
        #                                    use_bias=True
        #                                    )
        
        self.r_encoder = NR_GraphAttention(node_size=self.node_size,
                                           rel_size=rel_size,
                                           triple_size=triple_size,
                                           node_dim=self.input_dim,
                                           depth=self.depth,
                                           use_bias=True
                                           )

        if args.use_surface:
            self.name_fc = nn.Linear(768, output_dim)
            self.char_fc = nn.Linear(char_feature_dim, output_dim)
            torch.nn.init.xavier_uniform_(self.name_fc.weight)
            torch.nn.init.xavier_uniform_(self.char_fc.weight)
        
        '''
        Fusion Encoder
        '''
        self.fusion = RULEusion()
        

    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        
        return torch.sparse.mm(adj, emb)

    def forward(self,
                mask,
                img_features=None,
                att_features=None,
                name_features=None,
                char_features=None,
                adj_matrix=None,
                r_index=None,
                r_val=None,
                rel_adj=None,
                ent_adj=None
                ):

        #v1      
        if self.args.w_gcn and self.args.w_img and self.args.w_rel:
            ent_feature = self.avg(ent_adj, self.ent_embedding.weight, self.node_size)
            rel_feature = self.avg(rel_adj, self.rel_embedding.weight, self.rel_size)
            img_feature = self.img_fc(img_features)
            opt = [self.rel_embedding.weight, adj_matrix, r_index, r_val]
            rel_emb = self.r_encoder([rel_feature] + opt)
            gph_emb = self.e_encoder_cross([ent_feature] + opt+ [img_feature])
            img_emb = img_feature
            rel_emb = self.dropout(rel_emb)
            gph_emb = self.dropout(gph_emb)
            img_emb = self.dropout(img_emb)
        
        if self.args.w_attr and att_features is not None:
            att_emb = self.att_fc(att_features)
            att_emb = self.dropout(att_emb)
        else:
            att_emb = None
        
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
            name_emb = self.dropout(name_emb)
        else:
            name_emb = None
        
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
            char_emb = self.dropout(char_emb)
        else:
            char_emb = None
            

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb
