import torch
from torch import nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

class CustomMultiLossLayer(nn.Module):
    def __init__(self, loss_num):
        """
        Initializes the layer with the number of losses.
        
        Args:
            loss_num (int): Number of loss functions to be combined.
        """
        self.loss_num = loss_num
        super(CustomMultiLossLayer, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        """
        Combines multiple losses using learned weights.

        Args:
            loss_list (list): List of individual loss values.
        """
        self.loss_num = len(loss_list)
        # Calculate precision (inverse variance) for each loss
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            # Weight each loss by its precision and add a regularization term
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss
    
class DRL(nn.Module):
    """
    Dually Robust Loss (DRL) for Cross-KG Entity Alignment.
    
    This loss function implements Dually Robust Fusion with:
    - Uncertainty: measures the model's confidence in predictions
    - Consensus: measures the agreement between different modalities
    - L_DR: Dually Robust loss for robust training
    - L_Reg: KL divergence regularization loss
    """
    def __init__(self, tau=0.05, top_k=49, warmup_epoch=10, threshold=0.3, lambda2=0.0001):
        """
        Args:
            tau (float): Temperature parameter for evidence extraction.
            top_k (int): Number of top-k negative samples + 1 (including positive sample).
            warmup_epoch (int): Number of warmup epochs before applying robust training.
            threshold (float): Initial threshold value for sample division.
            lambda2 (float): Weight for KL divergence regularization (L_Reg).
        """
        super(DRL, self).__init__()
        self.tau = tau
        self.topk = top_k - 1
        self.thr = torch.tensor(threshold).cuda()
        self.warmup_epoch = warmup_epoch
        self.lambda2 = lambda2

    def KL(self, alpha, c):
        beta = torch.ones((1, c)).cuda()
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl
    
    def robust_ce_loss(self, alpha, label, lambda2=0.0001): 
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        alp = E * (1 - label) + 1
        B = lambda2 * self.KL(alp, label.size(1))
        return (A + B)   

    def robust_mse_loss(self, alpha, label, lambda2=0.0001):
        """Compute the Dually Robust MSE loss (L_DR) with KL regularization (L_Reg)."""
        S = torch.sum(alpha, dim=1, keepdim=True)
        m = alpha / S
        A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        # Use alpha-1 as evidence adjustment term
        E = alpha - 1
        alp = E * (1 - label) + 1
        C = lambda2 * self.KL(alp, label.size(1))
        return (A + B) + C
    
    def get_consensus_threshold(self):
        return self.threshold_consensus
    
    def get_uncertainty_threshold(self):
        return self.threshold_uncertainty

    def get_evidence(self, raw_sims):
        """Compute evidence, Uncertainty and normalized evidence from similarity matrix."""
        sims = torch.sigmoid(raw_sims)
        evidences = torch.exp(torch.tanh(raw_sims) / self.tau)
        sims_tanh = torch.tanh(raw_sims)
        sum_e = evidences
        norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
        alpha = evidences + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = evidences.size(1) / S
        return alpha, uncertainty, norm_e

    def get_true_positive_pairs(self, cosine_sims, margin=None):
        """Compute True Positive Pairs (TP) in both directions."""
        # Source to target direction
        diag_indices_i = torch.arange(cosine_sims.size(0), device=cosine_sims.device)
        max_indices_i = torch.argmax(cosine_sims, dim=1)
        TP_i = (max_indices_i == diag_indices_i)

        if margin is not None:
            sorted_sims_i, _ = torch.sort(cosine_sims, dim=1, descending=True)
            TP_i &= (sorted_sims_i[:, 0] - sorted_sims_i[:, 1]) > margin

        # Target to source direction
        diag_indices_j = torch.arange(cosine_sims.size(1), device=cosine_sims.device)
        max_indices_j = torch.argmax(cosine_sims.t(), dim=1)
        TP_j = (max_indices_j == diag_indices_j)

        if margin is not None:
            sorted_sims_j, _ = torch.sort(cosine_sims.t(), dim=1, descending=True)
            TP_j &= (sorted_sims_j[:, 0] - sorted_sims_j[:, 1]) > margin

        TP = torch.cat((TP_i, TP_j), dim=0).unsqueeze(1)
        return TP
    
    def get_weight(self, emb, train_links):
        """Compute Uncertainty and Consensus weights for embeddings."""
        with torch.no_grad():
            emb = F.normalize(emb, dim=1)
            zis = emb[train_links[:, 0]]
            zjs = emb[train_links[:, 1]]

            # Compute cosine similarity matrix
            cosine_sims = torch.mm(zis, zjs.t())
            # Extract positive sample similarities (diagonal elements)
            positive_i = torch.diag(cosine_sims).unsqueeze(1)
            positive_j = torch.diag(cosine_sims.t()).unsqueeze(1)

            # Mask diagonal positions as invalid
            sim_mask = torch.eye(cosine_sims.size(0), device=cosine_sims.device).bool()
            cosine_sims_no_diag = cosine_sims.masked_fill(sim_mask, float('-8'))

            # Select effective top-k count
            effective_k = min(self.topk, cosine_sims.size(1) - 1)
            # Get top-k negative sample similarities per row
            raw_sims_i, _ = torch.topk(cosine_sims_no_diag, k=effective_k, dim=1, largest=True)
            raw_sims_j, _ = torch.topk(cosine_sims_no_diag.t(), k=effective_k, dim=1, largest=True)

            # Insert positive sample similarity at first column
            raw_sims_i = torch.cat([positive_i, raw_sims_i], dim=1)
            raw_sims_j = torch.cat([positive_j, raw_sims_j], dim=1)

            # Get evidence (alpha) and Uncertainty from EDL
            alpha_i, uncertainty_i, _ = self.get_evidence(raw_sims_i)
            alpha_j, uncertainty_j, _ = self.get_evidence(raw_sims_j)

            # Concatenate Uncertainty from both directions
            uncertainty = torch.cat((uncertainty_i.detach(), uncertainty_j.detach()), dim=0)

            # Consensus from positive sample similarities, clamped to [0, 1]
            consensus = torch.cat((positive_i, positive_j), dim=0)
            consensus = torch.clamp(consensus, min=0.0, max=1.0)

        return uncertainty, consensus

    def forward(self, emb, train_links, mask=None, epoch=10, divide=False):
        """
        Compute the Dually Robust Loss with Uncertainty and Consensus.
        
        Args:
            emb (torch.Tensor): Entity or modality embeddings.
            train_links (torch.Tensor): Training entity pair indices.
            mask (torch.Tensor, optional): Mask for filtering valid samples.
            epoch (int): Current training epoch.
            divide (bool): Whether to process KG embeddings separately.
            
        Returns:
            loss: The combined L_DR (Dually Robust) and L_Reg (regularization) loss.
            uncertainty: Uncertainty scores for each sample.
            consensus: Consensus scores for each sample.
        """
        if divide:
            zis = emb[0]
            zjs = emb[1]
        else:
            emb = F.normalize(emb, dim=1)
            zis = emb[train_links[:, 0]]
            zjs = emb[train_links[:, 1]]

        cosine_sims = torch.mm(zis, zjs.t())
        positive_i = torch.diag(cosine_sims).unsqueeze(1)
        positive_j = torch.diag(cosine_sims.t()).unsqueeze(1)

        sim_mask = torch.eye(cosine_sims.size(0), device=cosine_sims.device).bool()
        cosine_sims_no_diag = cosine_sims.masked_fill(sim_mask, float('-8'))

        effective_k = min(self.topk, cosine_sims.size(1) - 1)
        raw_sims_i, _ = torch.topk(cosine_sims_no_diag, k=effective_k, dim=1, largest=True)
        raw_sims_j, _ = torch.topk(cosine_sims_no_diag.t(), k=effective_k, dim=1, largest=True)

        raw_sims_i = torch.cat([positive_i, raw_sims_i], dim=1)
        raw_sims_j = torch.cat([positive_j, raw_sims_j], dim=1)

        alpha_i, uncertainty_i, _ = self.get_evidence(raw_sims_i)
        alpha_j, uncertainty_j, _ = self.get_evidence(raw_sims_j)
        uncertainty = torch.cat((uncertainty_i.detach(), uncertainty_j.detach()), dim=0)

        with torch.no_grad():
            TP = self.get_true_positive_pairs(cosine_sims)
            consensus = torch.cat((positive_i, positive_j), dim=0)
            consensus = torch.clamp(consensus, min=0.0, max=1.0)

            if mask is not None:
                mask_cat = torch.cat((mask[train_links[:, 0]], mask[train_links[:, 1]])).bool()
                valid_TP = TP.squeeze() & mask_cat
                consensus_TP = consensus[valid_TP]
                uncertainty_TP = uncertainty[valid_TP]
                self.threshold_consensus = torch.max(self.thr, consensus_TP.min()) if consensus_TP.numel() > 0 else self.thr
                self.threshold_uncertainty = torch.min(1 - self.thr, uncertainty_TP.max()) if uncertainty_TP.numel() > 0 else 1 - self.thr
            else:
                consensus_TP = consensus[TP.squeeze()]
                uncertainty_TP = uncertainty[TP.squeeze()]
                self.threshold_consensus = torch.max(self.thr, consensus_TP.min()) if consensus_TP.numel() > 0 else self.thr
                self.threshold_uncertainty = torch.min(1 - self.thr, uncertainty_TP.max()) if uncertainty_TP.numel() > 0 else 1 - self.thr

            division_TP = ((consensus >= self.threshold_consensus) & (uncertainty < self.threshold_uncertainty)).squeeze()
            division_UFP = (uncertainty >= self.threshold_uncertainty).squeeze()
            division_IFP = ((consensus < self.threshold_consensus) & (uncertainty < self.threshold_uncertainty)).squeeze()

            if mask is not None:
                division_TP = division_TP & mask_cat
                division_UFP = division_UFP & mask_cat
                division_IFP = division_IFP & mask_cat

            y_i = torch.zeros(raw_sims_i.size(0), dtype=torch.long, device=raw_sims_i.device)
            y_j = torch.zeros(raw_sims_j.size(0), dtype=torch.long, device=raw_sims_j.device)
            prob_i = F.softmax(raw_sims_i, dim=1)
            prob_j = F.softmax(raw_sims_j, dim=1)
            prob = torch.cat((prob_i, prob_j), dim=0)
            label_i = F.one_hot(y_i, num_classes=prob_i.size(1))
            label_j = F.one_hot(y_j, num_classes=prob_i.size(1))
            TP_y = torch.cat((label_i, label_j), dim=0)
            IFP_y = consensus * TP_y + (1 - consensus) * prob

        alpha = torch.cat((alpha_i, alpha_j), dim=0)
        loss_TP = self.robust_mse_loss(alpha, TP_y, lambda2=self.lambda2)
        loss_IFP = self.robust_mse_loss(alpha, IFP_y, lambda2=self.lambda2)
        
        if epoch < self.warmup_epoch:
            loss = loss_TP.mean()
        else:
            loss = loss_TP[division_TP.squeeze()].mean() if torch.any(division_TP) else torch.tensor(0.0, device=loss_TP.device, requires_grad=True)
            if torch.any(division_IFP):
                loss = (loss + loss_IFP[division_IFP.squeeze()].mean()) / 2
            
        return loss, uncertainty, consensus

