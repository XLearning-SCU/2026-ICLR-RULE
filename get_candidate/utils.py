import os
import json
import torch
import numpy as np

def csls_sim(sim_mat, k, m_csls=2):
    """
    Compute pairwise CSLS similarity.
    """
    for i in range(m_csls):
        nearest_values1 = torch.mean(torch.topk(sim_mat, k)[0], 1)
        nearest_values2 = torch.mean(torch.topk(sim_mat.t(), k)[0], 1)
        sim_mat = 2 * sim_mat - nearest_values1.view(-1, 1) - nearest_values2.view(1, -1)
    return sim_mat

def read_file(file_paths):
    """Read alignment pair files (TSV format)."""
    tups = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                if len(params) >= 2:
                    tups.append(tuple([int(x) for x in params[:2]]))
    return tups

def analyze_metrics(distance, pairs, threshold, top_k_values=[1, 5, 10]):
    """Analyze metrics by splitting samples into high/low confidence based on threshold."""
    if not distance.is_cuda:
        distance = distance.cuda()

    n = distance.size(0)
    # Calculate max score for each row
    max_probs, _ = torch.max(distance, dim=1)

    high_mask = max_probs >= threshold
    low_mask = max_probs < threshold
    ground_truth = torch.arange(n, device=distance.device)

    def calc_ranks(sub_mask):
        subset_indices = torch.nonzero(sub_mask, as_tuple=True)[0]
        ranks = []
        if subset_indices.numel() == 0:
            return {f"top{k}": 0.0 for k in top_k_values}, 0.0, 0
            
        for idx in subset_indices:
            probs = distance[idx]
            sorted_indices = torch.argsort(probs, descending=True)
            rank = (sorted_indices == ground_truth[idx]).nonzero(as_tuple=True)[0]
            rank = rank[0].item() + 1 if rank.numel() > 0 else float('inf')
            ranks.append(rank)
        
        ranks_tensor = torch.tensor(ranks, device=distance.device, dtype=torch.float)
        metrics = {f"top{k}": (ranks_tensor <= k).sum().item() / len(ranks) * 100 for k in top_k_values}
        mrr = (1.0 / ranks_tensor).mean().item()
        return metrics, mrr, len(ranks)

    high_metrics, high_mrr, high_count = calc_ranks(high_mask)
    low_metrics, low_mrr, low_count = calc_ranks(low_mask)

    return {
        "high": {"metrics": high_metrics, "mrr": high_mrr, "count": high_count},
        "low":  {"metrics": low_metrics, "mrr": low_mrr, "count": low_count}
    }

def save_test_rank(test_distance, test_ill, output_path="", top_k=10):
    """Save test set candidate rankings to JSON."""
    global_indices = torch.tensor(test_ill[:, 1], device=test_distance.device)
    ranks = {}
    correct_in_top_k = 0
    total_samples = test_distance.size(0)

    for idx in range(total_samples):
        probs = test_distance[idx]
        sorted_indices = torch.argsort(probs, descending=True)[:top_k]
        sorted_sims = probs[sorted_indices].cpu().tolist()

        entity_id = int(test_ill[idx, 0])
        ref_id = int(test_ill[idx, 1])

        global_sorted_indices = global_indices[torch.argsort(probs, descending=True)]
        ground_rank = (global_sorted_indices == ref_id).nonzero()
        ground_rank = ground_rank[0].item() if ground_rank.numel() > 0 else -1

        if ground_rank < top_k:
            correct_in_top_k += 1

        ranks[entity_id] = {
            "ref": ref_id,
            "ground_rank": ground_rank,
            "candidates": [int(global_indices[j]) for j in sorted_indices],
            "cand_sims": [float(sim) for sim in sorted_sims]
        }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ranks, f, indent=4, ensure_ascii=False)

    correct_ratio = correct_in_top_k / total_samples if total_samples > 0 else 0.0
    print(f"Top-{top_k} correct: {correct_in_top_k}/{total_samples} ({correct_ratio:.2%})")
    print(f"Saved candidates to {output_path}")
