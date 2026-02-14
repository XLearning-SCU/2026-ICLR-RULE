"""
RULE: Robust Multi-Modal Entity Alignment with Uncertainty-aware Learning
Main training script for ICEWS dataset.
"""

import os
import os.path as osp
import gc
import pickle
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from torchlight.utils import set_seed
from config_ICEWS import cfg
from src.data_ICEWS import load_data, Collator_base
from src.utils import Loss_log, csls_sim
from model.RULE_ICEWS import RULE

logger = logging.getLogger(__name__)


class Runner:
    """Main training and evaluation runner for RULE model."""
    
    def __init__(self, args):
        self.args = args
        self.scaler = GradScaler()
        
        set_seed(args.random_seed)
        self._init_data()
        self._init_model()
        
        if self.args.only_test:
            self._init_dataloader(test_set=self.test_set)
        else:
            self._init_dataloader(
                train_set=self.train_set, 
                eval_set=self.eval_set, 
                test_set=self.test_set, 
                ent_set=self.ent_set
            )
            self._init_optimizer()
        
        self.best_performance = {'mrr': 0, 'metrics': None}

    def _init_data(self):
        """Initialize data from knowledge graphs."""
        self.KGs, _, self.train_set, self.test_set, \
            self.eval_set, _, self.ent_set = load_data(logger, self.args)
        self.eval_left = torch.LongTensor(self.test_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.test_set[:, 1].squeeze()).cuda()
        self.train_left = torch.LongTensor(self.train_set[:, 0].squeeze()).cuda()
        self.train_right = torch.LongTensor(self.train_set[:, 1].squeeze()).cuda()

    def _init_dataloader(self, train_set=None, eval_set=None, test_set=None, ent_set=None):
        """Initialize dataloaders for training and evaluation."""
        bs = self.args.batch_size
        collator = Collator_base(self.args)

        def create_dataloader(dataset, drop_last=False, shuffle=False):
            return dataset, DataLoader(
                dataset, num_workers=self.args.workers, persistent_workers=True,
                shuffle=shuffle, drop_last=drop_last, batch_size=bs, collate_fn=collator
            )

        if train_set is not None:
            _, self.train_dataloader = create_dataloader(
                train_set, drop_last=True, shuffle=True
            )
        if eval_set is not None:
            _, self.eval_dataloader = create_dataloader(eval_set)
        if test_set is not None:
            _, self.test_dataloader = create_dataloader(test_set)
        if ent_set is not None:
            _, self.ent_dataloader = create_dataloader(ent_set)

    def _init_model(self):
        """Initialize RULE model."""
        self.model = RULE(
            self.KGs, self.args, self.train_set, self.test_set, logger
        ).to(self.args.device)

    def _init_optimizer(self):
        """Initialize optimizer."""
        step_per_epoch = len(self.train_dataloader)
        self.args.total_steps = int(step_per_epoch * self.args.epoch)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def run(self):
        """Main training loop."""
        self.loss_log = Loss_log()
        self.curr_loss = 0.0
        self.step = 0
        self.curr_loss_dic = defaultdict(float)
        
        with tqdm(total=self.args.epoch) as pbar:
            for i in range(self.args.epoch):
                self.epoch = i
                self._train_epoch()
                
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                
                pbar.set_description(
                    f"Train | Ep [{self.epoch}/{self.args.epoch}] "
                    f"LR [{self.args.lr:.5f}] Loss {self.loss_item:.4f}"
                )
                pbar.update(1)
                self._update_loss_log()
                
                is_last = (i == self.args.epoch - 1)
                if (i + 1) % self.args.eval_epoch == 0 or is_last:
                    self.eval(last_epoch=is_last)

    def _process_batch(self, batch, batch_no=0):
        """Process a single training batch."""
        loss, output = self.model(batch, self.epoch, batch_no)
        
        self.scaler.scale(loss).backward()
        self.step += 1

        self._collect_statistics(loss, output)
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        torch.cuda.empty_cache()

        return loss.item()

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.loss_log.acc_init()
        
        for batch_no, batch in enumerate(self.train_dataloader):
            self._process_batch(batch, batch_no=batch_no)

    def _collect_statistics(self, loss, output):
        """Collect loss statistics for logging."""
        self.curr_loss += loss.item()
        if output is None:
            return
        
        if 'loss_dic' in output:
            for key, value in output['loss_dic'].items():
                self.curr_loss_dic[key] += value

    def _update_loss_log(self):
        """Update and log training losses."""
        for key, val in self.curr_loss_dic.items():
            if val != 0:
                logger.info(f"loss/{key}: {val:.4f}")

        self.curr_loss = 0.0
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.0

    def eval(self, last_epoch=False, save_name="", save_feature=False):
        """Evaluate the model."""
        self.model.eval()
        logger.info(" --------------------- Eval result --------------------- ")
        
        # Determine save paths for the last epoch
        if last_epoch:
            save_name = self._get_save_path(candidate=True)
            save_feature = True

        self._evaluate(
            self.eval_left, self.eval_right, 
            last_epoch=last_epoch, save_name=save_name, save_feature=save_feature
        )
        self.model.train()

    def test(self, save_name="", last_epoch=True):
        """Test the model."""
        if self.test_set is None:
            test_left, test_right = self.eval_left, self.eval_right
        else:
            test_left, test_right = self.train_left, self.train_right
        
        self.model.eval()
        logger.info(" --------------------- Test result --------------------- ")
        self._evaluate(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    def _evaluate(self, test_left, test_right, last_epoch=False, save_name="", save_feature=False):
        """Core evaluation logic."""
        with torch.no_grad():
            embs_dic = self.model.joint_emb_generat()
            final_emb = embs_dic.get('joint', None)
            
            if save_feature and save_name:
                save_dir = os.path.dirname(save_name)
                os.makedirs(save_dir, exist_ok=True)
                with open(save_name, 'wb') as f:
                    pickle.dump(final_emb, f)
                print(f"Features saved to {save_name}")

        # Compute similarity matrix
        distance = torch.matmul(final_emb[test_left], final_emb[test_right].transpose(0, 1))
        if self.args.csls:
            distance = csls_sim(distance, self.args.csls_k, self.args.m_csls)

        # Compute metrics
        top_k = [1, 5, 10]
        acc_l2r = np.zeros(len(top_k), dtype=np.float32)
        acc_r2l = np.zeros(len(top_k), dtype=np.float32)
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.

        for idx in range(test_left.shape[0]):
            _, indices = torch.sort(distance[idx, :], descending=True)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i, k in enumerate(top_k):
                if rank < k:
                    acc_l2r[i] += 1

        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=True)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i, k in enumerate(top_k):
                if rank < k:
                    acc_r2l[i] += 1

        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        
        for i in range(len(top_k)):
            acc_l2r[i] = round(float(acc_l2r[i] / test_left.size(0)), 4)
            acc_r2l[i] = round(float(acc_r2l[i] / test_right.size(0)), 4)
        
        gc.collect()
        
        logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.4f}, mrr = {mrr_l2r:.4f}")
        logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.4f}, mrr = {mrr_r2l:.4f}")

        best_mrr = max(mrr_l2r, mrr_r2l)
        
        if not self.args.only_test and best_mrr > self.best_performance['mrr'] and not last_epoch:
            logger.info(f"Best model update in Ep {self.epoch}: MRR from [{self.best_performance['mrr']:.4f}] --> [{best_mrr:.4f}]")
            self.best_performance = {
                'mrr': best_mrr,
                'metrics': {
                    'l2r': {'top1': float(acc_l2r[0]), 'top5': float(acc_l2r[1]), 'top10': float(acc_l2r[2]), 'mr': round(mean_l2r, 4), 'mrr': round(mrr_l2r, 4)},
                    'r2l': {'top1': float(acc_r2l[0]), 'top5': float(acc_r2l[1]), 'top10': float(acc_r2l[2]), 'mr': round(mean_r2l, 4), 'mrr': round(mrr_r2l, 4)}
                }
            }
            self._save_model(self.model, 'best')

        if last_epoch:
            self._save_model(self.model, 'last')
            logger.info(f"Best Performance: {self.best_performance['metrics']}")

    def _get_save_path(self, model_type="", candidate=False):
        """Generate save path based on configuration."""
        type_dir = 'prior_features' if candidate else 'save'
        save_dir = osp.join(self.args.data_path, type_dir, self.args.data_choice)
        os.makedirs(save_dir, exist_ok=True)
        
        setting = f"DNC_{self.args.eta}"
        if self.args.use_surface:
            setting += "_use_surface"

        suffix = f"_{model_type}" if not candidate else ""
        return osp.join(save_dir, f"{setting}_{self.args.data_split}{suffix}.pkl")

    def _save_model(self, model, model_type=""):
        """Save model checkpoint."""
        save_path = self._get_save_path(model_type)
        if model is None or not self.args.save_model:
            return
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'loss_mask': model.loss_mask
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saving [{save_path}] done!")


if __name__ == '__main__':
    # Parse configuration
    config = cfg()
    config.get_args()
    args = config.update_train_configs()
    
    set_seed(args.random_seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    args.device = torch.device(args.device)
    
    # Run training or testing
    runner = Runner(args)
    if args.only_test:
        runner.test(last_epoch=False)
    else:
        runner.run()
    
    logger.info("Training completed!")
