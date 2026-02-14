import torch
import random
import json
import numpy as np
import pdb
import os
import os.path as osp
from collections import Counter
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm

from .utils import get_topk_indices, get_adjr
import torch


class EADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Collator_base(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        # pdb.set_trace()

        return np.array(batch)


def load_data(logger, args):
    
    KGs, non_train, train_ill, test_ill, eval_ill, test_ill_,ent_ill = load_eva_data(logger, args)
    return KGs, non_train, train_ill, test_ill, eval_ill, test_ill_,ent_ill

def load_eva_data(logger, args):

    file_dir = osp.join(args.data_path, args.data_choice, args.data_split)
    lang_list = [1, 2]
    
    ent2id_dict, id2ent_dict, triples, adj_matrix, r_index, r_val, adj_features, rel_features = read_raw_data(file_dir, lang_list)
    
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(id2ent_dict)
    REL_NUM = max(r for _, r, _ in triples) + 1
    
    if args.noise_ratio > 0:
        img_vec_path = osp.join(
                args.data_path,
                "pkls",
                args.data_choice,
                args.data_split,
                f"image_train_noise_{args.eta:.1f}.pkl"
            )
    else:
        img_vec_path = osp.join(
                args.data_path,
                "pkls",
                args.data_choice,
                args.data_split, "img_feature.pkl")

    assert osp.exists(img_vec_path)
    
    word2vec_path = os.path.join(args.data_path, "embedding", "glove.6B.300d.txt")

    input_idx = torch.LongTensor(np.arange(ENT_NUM))

    img_features, missing_set = load_img(logger, ENT_NUM, img_vec_path)
    logger.info(f"image feature shape:{img_features.shape}")

    name_features = None
    char_features = None
    if args.data_choice == "DBP15K" and (args.w_name or args.w_char):

        if args.noise_ratio > 0:
            name_vec_path = osp.join(
                args.data_path,
                "pkls",
                args.data_choice,
                args.data_split,
                f"name_train_noise_{args.eta:.1f}.pkl"
            )
        else:
            name_vec_path = osp.join(
                    args.data_path,
                    "pkls",
                    args.data_choice,
                    args.data_split, "name_feature.pkl")

        assert osp.exists(word2vec_path)
        ent_vec, char_features = load_word_char_features(ENT_NUM, word2vec_path, args, logger)
        
        with open(name_vec_path, "rb") as f:
            features_name_dict = pickle.load(f)
            
        name_features = np.array(list(features_name_dict.values()), dtype=np.float32)
        
        logger.info(f"name feature shape:{name_features.shape}")
        logger.info(f"char feature shape:{char_features.shape}")
    
    train_ill = read_file([file_dir + "/sup_pairs"])
    train_ill = np.array(train_ill, dtype=np.int32)
    
    test_ill_ = read_file([file_dir + "/ref_pairs"])
    test_ill = np.array(test_ill_, dtype=np.int32)

    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))

    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

    logger.info(f"#left entity : {len(left_ents)}, #right entity: {len(right_ents)}")
    logger.info(f"#left entity not in train set: {len(left_non_train)}, #right entity not in train set: {len(right_non_train)}")

    a1 = os.path.join(file_dir, 'training_attrs_1')
    a2 = os.path.join(file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)  # attr
    logger.info(f"attribute feature shape:{att_features.shape}")

    logger.info("-----dataset summary-----")
    logger.info(f"dataset:\t\t {file_dir}")
    logger.info(f"triple num:\t {len(triples)}")
    logger.info(f"entity num:\t {ENT_NUM}")
    logger.info(f"relation num:\t {REL_NUM}")
    logger.info(f"train ill num:\t {train_ill.shape[0]} \t test ill num:\t {test_ill.shape[0]}")
    logger.info("-------------------------")

    eval_ill = None
    
    if args.noise_ratio > 0:
        shuffle_indices = np.random.choice(len(train_ill), int(len(train_ill) * args.eta), replace=False)  # Randomly select indices
        target_indices = train_ill[shuffle_indices, 1]  # Extract right-side entity indices
        shuffled_indices = np.random.permutation(target_indices)  # Shuffle indices
        train_ill[shuffle_indices, 1] = shuffled_indices  # Apply shuffled values
        
        # Merged logic from EA_NC (assuming train mode only, as test is not usually perturbed in training setup unless specified)
        ill_data = train_ill
        name_idx = np.random.choice(len(ill_data), int(len(ill_data) * args.eta), replace=False)
        img_idx = np.random.choice(len(ill_data), int(len(ill_data) * args.eta), replace=False)
        
        if args.use_surface:
            if args.data_choice == "DBP15K":
                name_features, char_features = shuffle_name_features(name_features, char_features, ill_data, name_idx)
            elif args.data_choice == "ICEWS":
                name_features = shuffle_features(name_features, ill_data, name_idx)
                
        img_features = shuffle_features(img_features, ill_data, img_idx)
                

    ills = np.concatenate((train_ill, test_ill), axis=0)
            
    # pdb.set_trace()
    train_ill = EADataset(train_ill)
    test_ill = EADataset(test_ill)
    ent_ill = EADataset(ills)
    
    
    return {
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,
        'id2ent': id2ent_dict,
        'ent2id': ent2id_dict,
        'images_list': img_features, 
        'missing_img':missing_set,
        'rel_features': rel_features,
        'att_features': att_features,
        'name_features': name_features,
        'char_features': char_features,
        'input_idx': input_idx, 
        "adj_matrix": adj_matrix,
        "r_index": r_index,
        "r_val": r_val,
        "adj_features": adj_features,
    }, {"left": left_non_train, "right": right_non_train}, train_ill, test_ill, eval_ill, test_ill_, ent_ill


def load_word2vec(path, dim=300):
    """
    glove or fasttext embedding
    """
    # print('\n', path)
    word2vec = dict()
    err_num = 0
    err_list = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="load word embedding"):
            line = line.strip('\n').split(' ')
            if len(line) != dim + 1:
                continue
            try:
                v = np.array(list(map(float, line[1:])), dtype=np.float64)
                word2vec[line[0].lower()] = v
            except:
                err_num += 1
                err_list.append(line[0])
                continue
    file.close()
    print("err list ", err_list)
    print("err num ", err_num)
    return word2vec


def load_char_bigram(path):
    """
    character bigrams of translated entity names
    """
    # load the translated entity names
    ent_names = json.load(open(path, "r"))
    # generate the bigram dictionary
    char2id = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in char2id:
                    char2id[word[idx:idx + 2]] = count
                    count += 1
    return ent_names, char2id


def load_word_char_features(node_size, word2vec_path, args, logger):
    """
    node_size : ent num
    """
    if args.AA_NC and args.eta!=0:
        namenoise_ratio > (args.data_path, "DBP15K", "translated_ent_name", "dbp_" + args.data_split + "_noise_" + str(args.eta) + ".json")
    else:
        name_path = os.path.join(args.data_path, "DBP15K", "translated_ent_name", "dbp_" + args.data_split + ".json")
    assert osp.exists(name_path)
    save_path_name = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_name.pkl")
    save_path_char = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_char.pkl")
    if osp.exists(save_path_name) and osp.exists(save_path_char):
        logger.info(f"load entity name emb from {save_path_name} ... ")
        ent_vec = pickle.load(open(save_path_name, "rb"))
        logger.info(f"load entity char emb from {save_path_char} ... ")
        char_vec = pickle.load(open(save_path_char, "rb"))
        return ent_vec, char_vec

    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    # generate the word-level features and char-level features

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(char2id)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, char2id[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(char2id)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    with open(save_path_name, 'wb') as f:
        pickle.dump(ent_vec, f)
    with open(save_path_char, 'wb') as f:
        pickle.dump(char_vec, f)
    logger.info("save entity emb done. ")
    return ent_vec, char_vec


def visual_pivot_induction(args, left_ents, right_ents, img_features, ills, logger):

    l_img_f = img_features[left_ents]  # left images
    r_img_f = img_features[right_ents]  # right images

    img_sim = l_img_f.mm(r_img_f.t())
    topk = args.unsup_k
    two_d_indices = get_topk_indices(img_sim, topk * 100)
    del l_img_f, r_img_f, img_sim

    visual_links = []
    used_inds = []
    count = 0
    for ind in two_d_indices:
        if left_ents[ind[0]] in used_inds:
            continue
        if right_ents[ind[1]] in used_inds:
            continue
        used_inds.append(left_ents[ind[0]])
        used_inds.append(right_ents[ind[1]])
        visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))
        count += 1
        if count == topk:
            break

    count = 0.0
    for link in visual_links:
        if link in ills:
            count = count + 1
    logger.info(f"{(count / len(visual_links) * 100):.2f}% in true links")
    logger.info(f"visual links length: {(len(visual_links))}")
    train_ill = np.array(visual_links, dtype=np.int32)
    return train_ill

def read_file(file_paths):
    """Read file content and return parsed tuple list."""
    tups = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = map(int, line.strip("\n").split("\t"))
                tups.append(tuple(params))
    return tups

def read_file_triplet(file_paths):
    """Read triplet or quintuple files and convert to triplet format."""
    tups = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                if len(params) == 3:
                    tups.append(tuple(map(int, params)))
                elif len(params) == 5:
                    tups.append(tuple(map(int, params[:3])))
                else:
                    print(len(params))
                    raise ValueError("Invalid number of columns in file: Expected 3 or 5 columns.")
    return tups

def read_raw_data(file_dir, lang=[1, 2]):
    """Read DBP15k/DWY15k dataset."""
    import numpy as np
    import scipy.sparse as sp

    print('loading raw data...')

    def read_dict(file_paths):
        """Read entity dictionary."""
        ent2id_dict = {}
        id2ent_dict = {}
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    eid, ename = line.strip("\n").split("\t")
                    eid = int(eid)
                    ent2id_dict[ename] = eid
                    id2ent_dict[eid] = ename
        return ent2id_dict, id2ent_dict

    def normalize_adj(adj):
        """Normalize adjacency matrix (symmetric normalization)."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        rowsum[rowsum == 0] = 1  # Prevent division by zero
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def normalize_features(features):
        """Row-normalize any matrix."""
        rowsum = np.array(features.sum(1)).flatten()  # Sum of each row
        rowsum[rowsum == 0] = 1  # Prevent division by zero
        inv_rowsum = np.power(rowsum, -1).flatten()
        d_inv = sp.diags(inv_rowsum)  # Build diagonal matrix
        return d_inv @ features

    # Read entity and triplet data
    ent2id_dict, id2ent_dict = read_dict([file_dir + f"/ent_ids_{i}" for i in lang])

    triples = read_file_triplet([file_dir + f"/triples_{i}" for i in lang])

    # Determine number of entities and relations
    ent_size = max(max(h, t) for h, _, t in triples) + 1
    rel_size = max(r for _, r, _ in triples) + 1

    # Initialize matrix data (build sparse matrix using coordinate lists)
    row, col, data = [], [], []
    rel_out = np.zeros((ent_size, rel_size))
    rel_in = np.zeros((ent_size, rel_size))
    radj = []

    # Build adjacency matrix and relation features
    for h, r, t in triples:
        row.extend([h, t])
        col.extend([t, h])
        data.extend([1, 1])  # values for bidirectional edges
        radj.append((h, t, r))
        radj.append((t, h, r + rel_size))
        rel_out[h, r] += 1
        rel_in[t, r] += 1

    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(ent_size, ent_size)).tocsr()

    # Build relation indices and values
    radj.sort(key=lambda x: (x[0], x[1]))  # Sort by tuple to improve processing efficiency
    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in radj:
        if (h, t) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add((h, t))
            r_index.append([count, r])
            r_val.append(1)
    r_val = [val / d[idx] for val, idx in zip(r_val, [i[0] for i in r_index])]

    # Build and normalize relation feature matrix
    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_matrix)  # Adjacency feature matrix
    rel_features = normalize_features(sp.csr_matrix(rel_features))  # Relation feature matrix

    return ent2id_dict, id2ent_dict, triples, adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features


def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    print('attr num:', len(fre))
    attr2id = {}
    # pdb.set_trace()
    topA = min(topA, len(fre))
    for i in range(topA):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr

def load_relation(e, KG, topR=1000):
    # (39654, 1000)
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict

def load_img(logger, e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    logger.info(f"{(100 * len(img_dict) / e_num):.2f}% entities have images")
    
    # init unknown img vector with zeros matrix
    img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0])  for i in range(e_num)])
    missing = [i for i in range(e_num) if i not in img_dict]
    logger.info(f"missing img features ratios: {100*len(missing)/e_num:.2f}%")
    return img_embd, missing

def shuffle_features(features, ill, shuffle_indices):
    """
    Shuffle the feature matrix for the selected indices.
    features: the feature matrix (n, d)
    ill: the current set of training or testing triples
    shuffle_indices: indices of the triples to shuffle
    """
    target_indices = ill[shuffle_indices, 1]  # Get target row indices
    shuffled_indices = np.random.permutation(target_indices)  # Shuffle indices
    features[target_indices] = features[shuffled_indices]  # Swap corresponding rows
    
    return features

def shuffle_name_features(name_features, char_features, ill, shuffle_indices):
    """
    Shuffle the feature matrices (name_features and char_features) for the selected indices in the same order.

    Args:
    - name_features (ndarray): Feature matrix for names (n*d).
    - char_features (ndarray): Feature matrix for characters (n*d).
    - ill (ndarray): The current set of training or testing triples.
    - shuffle_indices (ndarray): Indices of the triples to shuffle.

    Returns:
    - name_features (ndarray): Shuffled name features.
    - char_features (ndarray): Shuffled character features.
    """
    target_indices = ill[shuffle_indices, 1]  # Get target row indices
    shuffled_indices = np.random.permutation(target_indices)  # Shuffle indices
    
    name_features[target_indices] = name_features[shuffled_indices]
    char_features[target_indices] = char_features[shuffled_indices]

    return name_features, char_features
def l2_normalize(features):
    """L2 normalize features."""
    norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6  # prevent division by zero
    return features / norm

