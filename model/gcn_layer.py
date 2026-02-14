import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(src, index, dim=0, dim_size=None):
    index_expanded = index.unsqueeze(-1).expand_as(src)
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    return out.scatter_add_(dim, index_expanded, src)

class NR_GraphAttention(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(NR_GraphAttention, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()

        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]

        # features = self.activation(features)
outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]

            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                              size=[self.triple_size, self.rel_size], dtype=torch.float32)

            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0,:].long())

            features = new_features
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)

        return 
class NR_GraphAttentionCross(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(NR_GraphAttentionCross, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.attn_kernels_ent = nn.ParameterList()


        self.start_train =True


        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        features_c = inputs[5]

        # features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            attention_kernel_ent = self.attn_kernels_ent[l]

            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                              size=[self.triple_size, self.rel_size], dtype=torch.float32)
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)

            concat_fea = torch.cat([features_c[adj[0, :].long()], features_c[adj[1, :].long()]], dim=-1).unsqueeze(0)

            U, S, V = torch.pca_lowrank(concat_fea, center=True, q=self.node_dim)
            concat_ent = torch.matmul(concat_fea, V[:, :, :self.node_dim]).squeeze()

            concat_ent = torch.matmul(concat_ent, torch.diag_embed(torch.pow(S.squeeze() + 1e-5, -0.5)))
            concat_ent[torch.isinf(concat_ent)] = 0

            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            concat_ent = F.normalize(concat_ent, dim=1, p=2)
            neighs_rel = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel
            neighs_ent = neighs - 2*torch.sum(neighs*concat_ent, dim=1, keepdim=True)*concat_ent

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            att_ent = torch.squeeze(torch.mm(concat_ent, attention_kernel_ent), dim=-1)
            att_ent = torch.sparse_coo_tensor(indices=adj, values=att_ent, size=[self.node_size, self.node_size])
            att_ent = torch.sparse.softmax(att_ent, dim=1)

            new_features = scatter_sum(src=neighs_rel * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0, :].long())

            alpha = 0.1
            new_features = new_features + alpha * scatter_sum(src=neighs_ent * torch.unsqueeze(att_ent.coalesce().values(), dim=-1), dim=0,
                                                               index=adj[0, :].long())

            features = new_features
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)

        return 