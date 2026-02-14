"""
RULE: Robust Multi-Modal Entity Alignment with Uncertainty-aware Learning
Configuration for DBP15K dataset.
"""

import os.path as osp
import argparse


class cfg:
    """Configuration manager for RULE model on DBP15K dataset."""
    
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        self.data_root = osp.abspath(osp.join(self.this_dir))

    def get_args(self):
        parser = argparse.ArgumentParser(description='RULE for DBP15K Dataset')
        
        # ==================== Training Settings ====================
        parser.add_argument('--gpu', default=0, type=int, help='GPU device id')
        parser.add_argument('--batch_size', default=512, type=int, help='Training batch size')
        parser.add_argument('--epoch', default=40, type=int, help='Number of training epochs')
        parser.add_argument('--warmup_epoch', default=10, type=int, help='Warmup epochs for learning')
        parser.add_argument('--stop_reweight', default=25, type=int, help='Stop reweighting epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--clip', type=float, default=1.1, help='Gradient clipping threshold')
        parser.add_argument('--eval_epoch', default=1, type=int, help='Evaluate every n epochs')
        parser.add_argument('--random_seed', default=3408, type=int, help='Random seed for reproducibility')
        
        # ==================== Model Settings ====================
        parser.add_argument('--hidden_units', type=str, default='300,300,300',
                           help='Hidden units in each layer, comma-separated')
        parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
        parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
        parser.add_argument('--structure_encoder', type=str, default='gat', 
                           choices=['gat', 'gcn'], help='Structure encoder type')
        parser.add_argument('--heads', type=str, default='2,2', 
                           help='Attention heads in each GAT layer')
        parser.add_argument('--attr_dim', type=int, default=300, help='Attribute embedding dimension')
        parser.add_argument('--img_dim', type=int, default=300, help='Image embedding dimension')
        parser.add_argument('--char_dim', type=int, default=300, help='Character embedding dimension')
        
        # ==================== Loss Settings ====================
        parser.add_argument('--tau', type=float, default=0.07, help='Temperature for contrastive loss')
        parser.add_argument('--topk', type=int, default=50, help='Top-k for loss computation')
        parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for loss computation')
        parser.add_argument('--lambda2', type=float, default=0.0001, help='Regularization weight')
        
        # ==================== Data Settings ====================
        parser.add_argument('--data_choice', default='DBP15K', type=str, 
                           choices=['DBP15K', 'ICEWS'], help='Dataset choice')
        parser.add_argument('--data_split', default='zh_en', type=str,
                           choices=['zh_en', 'ja_en', 'fr_en'], help='Dataset split')
        parser.add_argument('--data_path', default='data', type=str, help='Data directory')
        parser.add_argument('--data_rate', type=float, default=0.3, help='Training data ratio')
        
        # ==================== Feature Settings ====================
        parser.add_argument('--use_surface', type=int, default=0, 
                           help='Use surface form features (name & char)')
        
        # ==================== Inference Settings ====================
        parser.add_argument('--csls', action='store_true', default=True, 
                           help='Use CSLS for inference')
        parser.add_argument('--csls_k', type=int, default=3, help='K for CSLS')
        parser.add_argument('--m_csls', default=2, type=int, help='M for CSLS')
        
        # ==================== Experiment Settings ====================
        parser.add_argument('--model_name', default='RULE', type=str, help='Model name')
        parser.add_argument('--model_name_save', default='', type=str, help='Model name for loading')
        parser.add_argument('--save_model', default=1, type=int, choices=[0, 1], help='Save model')
        parser.add_argument('--only_test', default=0, type=int, choices=[0, 1], help='Test only mode')
        parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
        parser.add_argument('--workers', type=int, default=12, help='Dataloader workers')
        
        # ==================== Noise Correspondence Settings ====================
        parser.add_argument('--noise_ratio', type=float, default=0.0, help='Noise ratio')
        
        self.cfg = parser.parse_args()

    def update_train_configs(self):
        """Update configuration based on settings."""
        self.cfg.eta = self.cfg.noise_ratio
        assert not (self.cfg.save_model and self.cfg.only_test), \
            "Cannot save model in test-only mode"

        self.cfg.data_root = self.data_root
        
        # Set feature flags
        self.cfg.w_gcn = True
        self.cfg.w_rel = True
        self.cfg.w_attr = True
        self.cfg.w_img = True
        self.cfg.w_name = bool(self.cfg.use_surface)
        self.cfg.w_char = bool(self.cfg.use_surface)
        
        # Build paths
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        
        # Dimension alias
        self.cfg.dim = self.cfg.attr_dim
        
        return self.cfg
