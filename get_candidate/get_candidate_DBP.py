import os
import pickle
import numpy as np
import torch
from utils import *

def get_feature_path(data_path, choice, split, eta, use_surface):
    """Constructs the feature file path matching the main script's naming convention."""
    setting = f"DNC_{eta}"
    if use_surface:
        setting += "_use_surface"
    filename = f"{setting}_{split}.pkl"
    return os.path.join(data_path, "prior_features", choice, filename)

def run(data_path, choice, split, eta, use_surface, threshold=0.2):
    feature_path = get_feature_path(data_path, choice, split, eta, use_surface)
    print(f"\n[INFO] Processing: {os.path.basename(feature_path)}")
    
    if not os.path.exists(feature_path):
        print(f"[WARN] File not found: {feature_path}\n")
        return

    with open(feature_path, 'rb') as f:
        features = pickle.load(f)

    # Load alignment pairs
    base_dir = os.path.join(data_path, choice, split)
    test_pair = np.array(read_file([os.path.join(base_dir, "ref_pairs")]), dtype=np.int32)

    def process_set(pairs, name):
        """Calculates distance and metrics for a given pair set."""
        if isinstance(features, torch.Tensor):
            left = features[pairs[:, 0]].float()
            right = features[pairs[:, 1]].float()
        else:
            left = torch.from_numpy(features[pairs[:, 0]]).float()
            right = torch.from_numpy(features[pairs[:, 1]]).float()
        
        # Calculate CSLS similarity
        dist = torch.matmul(left, right.t())
        dist = csls_sim(dist, k=3, m_csls=2)
        
        # Analyze metrics
        metrics_result = analyze_metrics(dist, pairs, threshold)
        
        print(f"{name} Set Results:")
        for subset, data in metrics_result.items():
            m = data['metrics']
            print(f"  {subset.capitalize()}: Top1 {m['top1']:.2f}% | Top5 {m['top5']:.2f}% | Top10 {m['top10']:.2f}% | MRR {data['mrr']:.4f} | Count {data['count']}")
        return dist

    print("-" * 60)
    test_dist = process_set(test_pair, "Test")
    
    # Save formatted results
    setting = f"DNC_{eta}" + ("_use_surface" if use_surface else "")
    output_dir = os.path.join("./candidate_json", choice, f"{setting}_{split}.json")
    save_test_rank(test_dist, test_pair, output_path=output_dir)
    print("-" * 60)

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "../data"
    CHOICE = "DBP15K"
    SPLITS = ["zh_en", "ja_en", "fr_en"]
    ETAS = [0.0, 0.2, 0.5]
    USE_SURFACE = False
    THRESHOLD = 0.2

    for split in SPLITS:
        for eta in ETAS:
            run(DATA_PATH, CHOICE, split, eta, USE_SURFACE, THRESHOLD)
