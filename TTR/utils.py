import os
import json
import re  # 确保导入正则表达式模块
import torch
import pandas as pd

### json file IO
def load_json(path):
    with open(path, "r", encoding="utf-8") as fr:
        datas = json.load(fr)
    return datas

def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, ensure_ascii=False, indent=4)


### tool functions
def merge_dict(dic1, dic2):
    return {**dic1, **dic2}

def transform_idx_to_int(dic:dict):
    return {int(idx):data for idx, data in dic.items()}

def count_ranks(ranks):
    count = [0, 0, 0, 0]
    span = ['[ 0,  1)', '[ 1, 10)', '[10, 20)', '[20, --)']
    for r in ranks:
        if r == 0:
            count[0] += 1
        else:
            if r < 10:
                count[1] += 1
            else:
                if r < 20:
                    count[2] += 1
                else:
                    count[3] += 1
    total = len(ranks)
    print('Count of Ranks: ')
    for i in range(len(count)):
        print(f'  {span[i]} : {count[i]} , {count[i] / total:.2%}')


### evaluate
def evaluate_alignment(ranks, hit_k=[1, 5, 10]):
    hits = [0] * len(hit_k)
    mrr = 0
    for r in ranks:
        mrr += 1 / (r + 1)
        for j in range(len(hit_k)):
            if r < hit_k[j]:
                hits[j] += 1
    total_num = len(ranks)
    mrr /= total_num
    hits = [round(hits[i] / total_num, 4) for i in range(len(hit_k))]
    
    return hits, mrr


### generate entity neighbors information
class NeighborGenerator(object):
    def __init__(self, cand_file, data_file_path):
        self.cand_file = cand_file
        self.data_file_path = data_file_path
        self.ref, self.rank, self.cand, self.cand_score = self.load_candidates()
        # print(self.cand)
        self.entities = sorted([int(e) for e in self.cand.keys()])
        self.ent_name = self.load_name_dict()
        
    def load_name_dict(self):
        with open(os.path.join(self.data_file_path, 'candidates', 'name_dict'), 'r', encoding='utf-8') as fr:
            name_dict = json.load(fr)
        ent_name = transform_idx_to_int(name_dict['ent'])
        return ent_name
    
    # initialize, load data
    def load_candidates(self):
        with open(self.cand_file, 'r', encoding='utf-8') as fr:
            origin_cand = json.load(fr)
        ref, rank, cand, cand_score = {}, {}, {}, {}
        for eid, data in origin_cand.items():
            eid = int(eid)
            ref[eid] = data['ref']
            rank[eid] = data['ground_rank']
            cand[eid] = data['candidates']
            cand_score[eid] = data['cand_sims']
        return ref, rank, cand, cand_score

    # API
    def get_all_entities(self):
        all_ent = set()
        for eid, cand in self.cand.items():
            all_ent.update([eid] + cand)
        return sorted(list(all_ent))

    def get_entities(self):
        return self.entities
    
    def get_ref_ent(self, ent_id:int):
        return self.ref[ent_id]
    
    def get_base_rank(self, ent_id:int):
        return self.rank[ent_id]
    
    def get_candidates(self, ent_id:int):
        cand = []
        for score, cand_id in zip(self.cand_score[ent_id], self.cand[ent_id]):
            cand_ent = {"ent_id": cand_id}
            cand_ent["hhea_sim"] = round(score, 3)
            cand_ent["img_path"] = os.path.join(self.data_file_path, "concat_images", f"{cand_id}.jpg")
            cand_ent["name"] = self.ent_name[cand_id]
            cand.append(cand_ent)
        return cand
    
    def get_main_entity(self, ent_id: int):
        main_entity = {"ent_id": ent_id, "name": self.ent_name[ent_id]}
        main_entity["img_path"] = os.path.join(self.data_file_path, "concat_images", f"{ent_id}.jpg")
        return main_entity


# def get_score(res: str, field: str = "NAME SIMILARITY"):
#     content = res.replace('\n', ' ').replace('\t', ' ')
#     content = re.sub(r'\d{2}\d*', '', content)
#     score = 0
#     if field in content:
#         score_find = re.findall(f"{field}\D*[=|be|:] \d+", content)
#         if len(score_find) > 0:
#             score_find = re.findall(f"\d+", score_find[-1])
#             score = int(score_find[-1])
#     if score < 0:
#         print(f'#### SCORE ERROR : {score}')
#         score = 0
#     if score > 10:
#         print(f'#### SCORE ERROR : {score}')
#         score = 10
#     return score

def get_score(res: str):
    content = res.replace('\n', ' ').replace('\t', ' ')
    score = 0

    # 直接匹配 "= X out of 10"，前面可以是任意字符
    match = re.search(r"=\s*(\d+)\s*out of 10", content)

    if match:
        score = int(match.group(1))  # 提取 X
        score = max(0, min(score, 10))  # 限制范围 0-10

    return score

def get_uncertainty(scores, tau=1):
    """
    计算不确定性 U，以及 evidence-based 的参数 alpha，
    基于 Evidence-based Deep Learning (EDL) 的思路。

    参数:
        scores: torch.Tensor，形状可以为 [n] 或 [n, d]，
                如果 scores 是一维，则认为每个元素代表一个类别，总类别数为 n。
        tau: float，超参数，用于调节 tanh 后的指数计算。

    返回:
        uncertainty: float，不确定性 U，定义为 (类别数 / alpha 的和)
    """
    # 1. 对 scores 应用 tanh
    scores_tanh = torch.tanh(scores)

    # 2. 计算 evidence：exp(tanh(scores)/tau)
    evidence = torch.exp(scores_tanh / tau)

    # 3. 计算 alpha: evidence + 1
    alpha = evidence + 1

    num_categories = scores.shape[0]

    # 5. 计算 S，即所有 alpha 的和
    S = torch.sum(alpha)

    # 6. 计算不确定性 U：类别数 / S
    uncertainty = num_categories / S

    return uncertainty.item()

def save_results_to_excel(args, last_performance, save_dir="result"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.data_choice + "_" + args.data_split + ".xlsx")
    
    # 只提取关键的 args 参数
    selected_settings = ["use_surface", "eta", "sigma"]
    args_dict = {key: getattr(args, key, None) for key in selected_settings}
    
    # 处理 last_performance
    last_metrics = last_performance.get("metrics", {})
    last_data = {**args_dict}
    for metric_type, metrics in last_metrics.items():
        for key, val in metrics.items():
            last_data[f"{metric_type}_{key}"] = val
    df_last = pd.DataFrame([last_data])
    
    # 检查文件是否存在
    if os.path.exists(save_path):
        with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing_last = pd.read_excel(save_path, sheet_name="last")
            df_last = pd.concat([existing_last, df_last], ignore_index=True)
            df_last.to_excel(writer, sheet_name="last", index=False)
    else:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_last.to_excel(writer, sheet_name="last", index=False)
    
    print(f"Performance results saved to {save_path}")