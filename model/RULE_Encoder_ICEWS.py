from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .Tool_model import GAT, GCN


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
                 char_feature_dim=None):
        super(RULE_Encoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        clip_name_dim = self.args.clip_name_dim
        char_dim = self.args.char_dim
        self.ENT_NUM = ent_num

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        '''
        Entity Embedding
        '''
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True
        
        '''
        Modal Encoder
        '''
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        
        torch.nn.init.xavier_uniform_(self.img_fc.weight)
        torch.nn.init.xavier_uniform_(self.rel_fc.weight)
       
        if self.args.w_name and self.args.w_char:
            self.name_fc = nn.Linear(clip_name_dim, char_dim)
            self.char_fc = nn.Linear(char_feature_dim, char_dim)
            torch.nn.init.xavier_uniform_(self.name_fc.weight)
            torch.nn.init.xavier_uniform_(self.char_fc.weight)
        
        args.dropout=self.args.dropout
        args.attn_dropout=self.args.attn_dropout
        self.dropout = nn.Dropout(self.args.dropout) #0.3 wiki 0.4 yago
        
        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=False, diag=True)
        '''
        Fusion Encoder
        '''
        self.fusion = RULEusion()


    def forward(self,
                input_idx,
                adj,
                mask,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None
                ):

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
            gph_emb = self.dropout(gph_emb)
        else:
            gph_emb = None
        
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
            img_emb = self.dropout(img_emb)
        else:
            img_emb = None
        
        if self.args.w_rel and rel_features is not None:
            rel_emb = self.rel_fc(rel_features)
            rel_emb = self.dropout(rel_emb)
        else:
            rel_emb = None
        
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
