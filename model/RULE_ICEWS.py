import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .RULE_Loss import CustomMultiLossLayer, DRL
from .RULE_Encoder_ICEWS import RULE_Encoder
from src.utils import pairwise_distances
import itertools
import math

class RULE(nn.Module):
    """RULE: Robust Multi-Modal Entity Alignment with Uncertainty-aware Learning."""
    
    def __init__(self, kgs, args, train_set, test_set, logger):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.logger = logger
        self.test_ill = test_set
        self.train_ill = train_set
        self.cross_modal_key = {}
        self.transfer_label = {}
        
        self.missing_img = self.kgs['missing_img']
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()

        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = None        
        self.att_features = None
        self.name_features = None
        self.char_features = None
        
        if kgs["name_features"] is not None:
            self.name_features = F.normalize(torch.FloatTensor(kgs["name_features"])).cuda()
            
        if kgs["char_features"] is not None:
            self.char_features = kgs["char_features"].cuda()
        
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
            
        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.modal_keys = ['image', 'structure', 'relation', 'attribute', 'name', 'char']
        self.loss_mask = {}
        
        for key in self.modal_keys:
            self.loss_mask[key] = torch.ones(kgs['ent_num'], dtype=torch.float).cuda()
            
        for i in self.missing_img:
            self.loss_mask['image'][i] = 0

        self.multimodal_encoder = RULE_Encoder(args=self.args,
                                            ent_num=kgs["ent_num"],
                                            img_feature_dim=img_dim,
                                            char_feature_dim=char_dim)
        
        # Compute number of modalities dynamically
        self.modality_num = 2
        if self.args.use_surface:
            self.modality_num += 1
        
        if self.modality_num == 2:
            self.consensus_modality_num = 1
        else:
            self.consensus_modality_num = math.floor(self.modality_num / 2) + 1 #math.ceil(self.modality_num / 2)

        self.multi_loss_layer = CustomMultiLossLayer(self.modality_num)
       
        self.robust_loss = DRL(tau=self.args.tau, top_k=self.args.topk, warmup_epoch = self.args.warmup_epoch, threshold = self.args.threshold, lambda2=self.args.lambda2)
        

    def joint_emb_generat(self):
        emb_dict = self.emb_generat()
        gph_emb = emb_dict.get('structure', None)
        img_emb = emb_dict.get('image', None)
        rel_emb = emb_dict.get('relation', None)
        att_emb = emb_dict.get('attribute', None)
        name_emb = emb_dict.get('name', None)
        char_emb = emb_dict.get('char_name', None)

        emb_dict =  {"structure": gph_emb, "image":img_emb, "relation":rel_emb, "attribute": att_emb, "name": name_emb,\
                    "char": char_emb}
        
        emb_dict = {key: F.normalize(value) for key, value in emb_dict.items() if value is not None}   

        joint_emb = self.multimodal_encoder.fusion(emb_dict, self.loss_mask)

        # return {"joint": joint_emb}
        return {"structure": gph_emb, "image":img_emb, "relation":rel_emb, "attribute": att_emb, "name": name_emb,\
                    "char": char_emb, "joint": joint_emb}


    def forward(self, batch, epoch, batch_no):
        self.loss_dic = {}
        self.epoch = epoch
        
        emb_dict = self.emb_generat()
        gph_emb = emb_dict.get('structure', None)
        img_emb = emb_dict.get('image', None)
        rel_emb = emb_dict.get('relation', None)
        att_emb = emb_dict.get('attribute', None)
        name_emb = emb_dict.get('name', None)
        char_emb = emb_dict.get('char_name', None)

        emb_dict =  {"structure": gph_emb, "image":img_emb, "relation":rel_emb, "attribute": att_emb, "name": name_emb,\
                    "char": char_emb}
        
        emb_dict = {key: F.normalize(value) for key, value in emb_dict.items() if value is not None}     
            
        loss_attribute, uncertainty, consensus = self.crossgraph_attribute_alignment(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch, epoch)
        
        if epoch >= self.args.warmup_epoch and epoch <= self.args.stop_reweight: 
            joint_emb = self.multimodal_encoder.fusion(emb_dict, self.loss_mask)
            uncertainty_joint, consensus_joint = self.robust_loss.get_weight(joint_emb, batch)
            self.DRF_train(emb_dict, batch, uncertainty, consensus, uncertainty_joint, consensus_joint)
            
        if epoch >= self.args.warmup_epoch and batch_no == 0 and epoch % 1==0 and epoch <= self.args.stop_reweight: 
            self.DRF_test(self.test_ill, emb_dict)

            
        joint_emb = self.multimodal_encoder.fusion(emb_dict, self.loss_mask)

        loss_entity, _, _ = self.robust_loss(joint_emb, batch, epoch=epoch)
        
        loss_all = loss_attribute + loss_entity
        self.loss_dic.update({"loss_all": loss_all, "loss_entity": loss_entity, "loss_attribute": loss_attribute})
        
        output = {"loss_dic": self.loss_dic,"loss_all": loss_all, "gph_emb": gph_emb, "img_emb":img_emb, "rel_emb": rel_emb, \
            "att_emb": att_emb, "name_emb": name_emb, "char_emb":char_emb,"joint_emb": joint_emb}
        
        return  loss_all, output
 
    def crossgraph_attribute_alignment(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, epoch):
        embeddings = {
            'structure': gph_emb,
            'relation': rel_emb,
            'attribute': att_emb,
            'image': img_emb,
            'name': name_emb,
            'char': char_emb,
        }

        losses = {}
        uncertainty = {}
        consensus = {}

        # Compute loss and uncertainty for each modality
        for key, emb in embeddings.items():
            if emb is not None:
                losses[key], uncertainty[key], consensus[key] = self.robust_loss(emb, train_ill, mask=self.loss_mask[key], epoch=epoch)
            else:
                losses[key], uncertainty[key], consensus[key] = 0, None, None
                

        # Determine which loss dictionary to update based on train_ill size
        loss_key_prefix = "test_" if len(train_ill) > self.args.batch_size else ""
        self.loss_dic.update({f"{loss_key_prefix}{key}_loss": loss for key, loss in losses.items()})

        # Filter out zero losses for multi-loss computation
        valid_losses = [loss for loss in losses.values() if loss != 0]
        total_loss = self.multi_loss_layer(valid_losses)

        return total_loss, uncertainty, consensus
    
    def emb_generat(self):
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb = \
                                    self.multimodal_encoder(self.input_idx,
                                                            self.adj,
                                                            self.loss_mask,
                                                            self.img_features,
                                                            self.rel_features,
                                                            self.att_features,
                                                            self.name_features,
                                                            self.char_features)
    
        return {"structure": gph_emb, "image":img_emb, "relation":rel_emb, "attribute": att_emb, "name": name_emb,\
            "char_name": char_emb}
    
    def DRF_train(self, emb_dict, batch, uncertainty, consensus, uncertainty_joint, consensus_joint):
        with torch.no_grad():
            # Compute W_joi with shape (2b, 1)
            # uncertainty_joint was U_joi, consensus_joint was TRS_joi
            W_joi = (1 - uncertainty_joint) / 2 + consensus_joint / 2
            
            # Only update loss_mask for samples with W_joi >= 0.5
            W_joi_mask = (W_joi.view(-1) >= 0.5)
            
            # Iterate through modalities to update loss_mask
            for key in self.modal_keys:
                if emb_dict.get(key, None) is not None:
                    # Get Uncertainty and Consensus for current modality
                    batch_uncertainty = uncertainty.get(key).view(-1)
                    batch_consensus = consensus.get(key).view(-1)
                    
                    batch_certainty = 1 - batch_uncertainty  # Compute certainty

                    # First part of batch (first b samples)
                    b = batch.shape[0]  # Assuming batch.shape[0] is b
                    mask_first = W_joi_mask[:b]  # Update mask for first b samples
                    indices_first = torch.tensor(batch[:, 0], dtype=torch.long).cuda()[mask_first]
                    if indices_first.numel() > 0:
                        self.loss_mask[key][indices_first] = (
                            batch_certainty[:b][mask_first] / 2 +
                            batch_consensus[:b][mask_first] / 2
                        )
                        
                    # Second part of batch (last b samples)
                    mask_second = W_joi_mask[b:]
                    indices_second = torch.tensor(batch[:, 1], dtype=torch.long).cuda()[mask_second]
                    if indices_second.numel() > 0:
                        self.loss_mask[key][indices_second] = (
                            batch_certainty[b:][mask_second] / 2 +
                            batch_consensus[b:][mask_second] / 2
                        )
    
    def DRF_test(self, test_links, emb_dic):
        with torch.no_grad():
            self.logger.info(f"---------modality_freezing--------")
            
            # Initialize test_sim_dic to store test_sim matrices for each freeze_key
            test_sim_dic = {}
            for freeze_key in self.modal_keys:
                if freeze_key not in emb_dic:
                    continue

                # Normalize embedding vectors
                norm_emb = F.normalize(emb_dic[freeze_key])
                # Compute test_sim matrix
                test_sim = norm_emb[test_links[:, 0]] @ norm_emb[test_links[:, 1]].t()
                
                # Handle missing images
                if freeze_key == "image":
                    # Split test_links into two columns
                    test_links_0 = torch.tensor(test_links[:, 0], dtype=torch.long).cuda()
                    test_links_1 = torch.tensor(test_links[:, 1], dtype=torch.long).cuda()

                    # Convert self.missing_img
                    missing_img_tensor = torch.tensor(self.missing_img, dtype=torch.long).cuda()

                    # Generate missing_mask
                    missing_mask = torch.isin(test_links_0, missing_img_tensor) | torch.isin(test_links_1, missing_img_tensor)

                    # Zero out rows and columns for missing IDs
                    test_sim[missing_mask, :] = 0  
                    test_sim[:, missing_mask] = 0  
                        
                # Stack matrices vertically using torch.cat (asymmetric)
                test_sim_dic[freeze_key] = torch.cat((test_sim, test_sim.t()), dim=0)

                # Initialize max similarity matrix
            max_sim_matrix = None

            # Step 2: Compute average similarity for modality combinations
            modal_keys = list(test_sim_dic.keys())
            for r in range(self.consensus_modality_num, len(modal_keys) + 1):
                for comb in itertools.combinations(modal_keys, r):
                    # Initialize zero matrix for accumulating combined similarity
                    avg_sim = torch.zeros_like(next(iter(test_sim_dic.values())))
                    for key in comb:
                        avg_sim += test_sim_dic[key].clone()  # Accumulate current modality combination similarity
                    avg_sim /= len(comb)  # Compute average similarity

                    # Update max similarity matrix
                    if max_sim_matrix is None:
                        max_sim_matrix = avg_sim
                    else:
                        max_sim_matrix = torch.maximum(max_sim_matrix, avg_sim)
            
            # Now find the final TP
            TP = torch.argmax(max_sim_matrix, dim=1)

            for freeze_key in self.modal_keys:
                if freeze_key not in emb_dic:
                    continue
                norm_emb = F.normalize(emb_dic[freeze_key])
                
                test_sim = norm_emb[test_links[:,0]] @ norm_emb[test_links[:,1]].t()
                
                if freeze_key == "image":
                    # Split test_links into two columns
                    test_links_0 = torch.tensor(test_links[:, 0], dtype=torch.long).cuda()
                    test_links_1 = torch.tensor(test_links[:, 1], dtype=torch.long).cuda()

                    # Convert self.missing_img
                    missing_img_tensor = torch.tensor(self.missing_img, dtype=torch.long).cuda()

                    # Generate missing_mask
                    missing_mask = torch.isin(test_links_0, missing_img_tensor) | torch.isin(test_links_1, missing_img_tensor)

                    # Zero out rows and columns for missing IDs
                    test_sim[missing_mask, :] = 0  
                    test_sim[:, missing_mask] = 0 
                
                test_sim = torch.cat((test_sim,test_sim.t()),dim=0)
                
                test_sim_topk, _ = torch.topk(test_sim, k=50, dim=1, largest=True)
                
                _, test_uncertainty, _ = self.robust_loss.get_evidence(test_sim_topk)
                
                test_certainty = (1 - test_uncertainty)

                consensus = test_sim[torch.arange(test_sim.size(0)), TP]
                consensus = torch.clamp(consensus, min=0.0, max=1.0)
                
                # original
                self.loss_mask[freeze_key][test_links[:,0]] = test_certainty[:len(test_links[:,0]),0]/2 + consensus[:len(test_links[:,0])]/2 
                self.loss_mask[freeze_key][test_links[:,1]] = test_certainty[len(test_links[:,0]):,0]/2 + consensus[len(test_links[:,0]):]/2 
        
            for i in self.missing_img:
                self.loss_mask['image'][i] = 0

