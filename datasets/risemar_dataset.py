import os
import sys
sys.path.append('..')
import json
import torch
import warnings
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utilities.ct_tools import CTTransforms
from utilities.misc import rotate_tensor


class RiseMARDataset(Dataset):
    """
    A dataset class that loads and processes CT images from different CT data sources.
    In the original RISE-MAR paper, the sources include DeepLesion dataset, private 
    Dental dataset, and CTPelvic1K dataset. 
    
    The class supports both paired and unpaired data loading, with options for:
    - Train/val/test splitting
    - Data augmentation (random flips and rotations)
    - HU value normalization
    - Configurable train/val ratios
    
    Each dataset can be loaded **independently** by providing the corresponding JSON 
    metadata file. The JSON files should contain paths and metadata for the CT images. 
    Examples for the JSON file can be found in `./data/meta/example_*.json`
    """
    def __init__(
        self, 
        deepl_json=None,  # JSON path to the metadata for paired DeepLesion dataset
        dental_json=None,  # JSON path to the metadata for paired Dental dataset
        pelvic_json=None,  # JSON path to the metadata for paired CTPelvic1K dataset
        unpaired_json1=None,  # JSON path to the metadata for unpaired dataset loaded along with paired DeepLesion
        unpaired_json2=None,  # JSON path to the metadata for unpaired dataset loaded along with paired Dental
        unpaired_json3=None,  # JSON path to the metadata for unpaired dataset loaded along with paired CTPelvic1K
        mode='train',   # training mode
        num_train=0.9,  # number of cases or split-ratio for the training set
        num_val=0.1,    # number of cases or split-ratio for the validation set
        min_hu=-1024,   # minimum HU value
        max_hu=3072,    # maximum HU value 
        seed=3407,      # random seed
        flip_prob=0,    # probability for random flipping
        rot_prob=0,     # probability for random rotation
    ):
        super().__init__()
        assert mode in ['train', 'val', 'test'], f"Unsupported training mode: {mode}."
        assert num_train is not None
        assert num_val is not None
        used_datasets = []
        if deepl_json is not None:
            used_datasets.append('deepl')
        if dental_json is not None:
            used_datasets.append('dental')
        if pelvic_json is not None:
            used_datasets.append('pelvic')
        
        self.deepl_json = deepl_json  
        self.dental_json = dental_json  
        self.pelvic_json = pelvic_json  
        self.unpaired_json1 = unpaired_json1  
        self.unpaired_json2 = unpaired_json2  
        self.unpaired_json3 = unpaired_json3  
        
        self.mode = mode
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.num_train = num_train
        self.num_val = num_val
        self.seed = seed
        self.flip_prob = flip_prob if mode == 'train' else 0
        self.rot_prob = rot_prob if mode == 'train' else 0
        
        np.random.seed(self.seed)
        self.cttool = CTTransforms(min_hu=min_hu, max_hu=max_hu)
        self.num_datasets = len(used_datasets)
        
        self.deepl_keys, self.deepl_meta = self.prepare_paired_data(self.deepl_json)
        self.dental_keys, self.dental_meta = self.prepare_paired_data(self.dental_json)
        self.pelvic_keys, self.pelvic_meta = self.prepare_paired_data(self.pelvic_json)
        self.unpaired_ma_orders1, self.unpaired_ma_keys1, self.unpaired_ma_meta1 = self.prepare_unpaired_data(self.unpaired_json1, key='ma')
        self.unpaired_mf_orders1, self.unpaired_mf_keys1, self.unpaired_mf_meta1 = self.prepare_unpaired_data(self.unpaired_json1, key='mf')
        self.unpaired_ma_orders2, self.unpaired_ma_keys2, self.unpaired_ma_meta2 = self.prepare_unpaired_data(self.unpaired_json2, key='ma')
        self.unpaired_mf_orders2, self.unpaired_mf_keys2, self.unpaired_mf_meta2 = self.prepare_unpaired_data(self.unpaired_json2, key='mf')
        self.unpaired_ma_orders3, self.unpaired_ma_keys3, self.unpaired_ma_meta3 = self.prepare_unpaired_data(self.unpaired_json3, key='ma')
        self.unpaired_mf_orders3, self.unpaired_mf_keys3, self.unpaired_mf_meta3 = self.prepare_unpaired_data(self.unpaired_json3, key='mf')
        
        print(
        f"Initializing CQADataset (mode={mode}):"
        f"- Used datasets: {used_datasets}, "
        f"- Paired number: {len(self.deepl_keys)}/{len(self.dental_keys)}/{len(self.pelvic_keys)},"
        f"- Effective length: {self.__len__()},"
        f"- Unpaired MA number: {len(self.unpaired_ma_keys1)}/{len(self.unpaired_ma_keys2)}/{len(self.unpaired_ma_keys3)},"
        f"- Unpaired MF number: {len(self.unpaired_mf_keys1)}/{len(self.unpaired_mf_keys2)}/{len(self.unpaired_mf_keys3)},"
        f"- Probabilities for random flip/rotation: {self.flip_prob}/{self.rot_prob}."
        )
    
    def __len__(self):
        return max(len(self.dental_keys), len(self.deepl_keys), len(self.pelvic_keys)) * self.num_datasets
    
    def __getitem__(self, index):
        n = self.num_datasets
        if index % n == 0:
            data = self.get_paired_and_unpaired_items(
                index, self.deepl_keys, self.deepl_meta,
                self.unpaired_ma_orders1, self.unpaired_ma_keys1, self.unpaired_ma_meta1,
                self.unpaired_mf_orders1, self.unpaired_mf_keys1, self.unpaired_mf_meta1)
            dataset_flag = 0
                
        elif index % n == 1:
            data = self.get_paired_and_unpaired_items(
                index, self.dental_keys, self.dental_meta,
                self.unpaired_ma_orders2, self.unpaired_ma_keys2, self.unpaired_ma_meta2,
                self.unpaired_mf_orders2, self.unpaired_mf_keys2, self.unpaired_mf_meta2)
            dataset_flag = 1
                
        else:
            data = self.get_paired_and_unpaired_items(
                index, self.pelvic_keys, self.pelvic_meta,
                self.unpaired_ma_orders3, self.unpaired_ma_keys3, self.unpaired_ma_meta3,
                self.unpaired_mf_orders3, self.unpaired_mf_keys3, self.unpaired_mf_meta3)
            dataset_flag = 2
        
        data.update({'dataset_flag': dataset_flag})
        return data

    
    def get_paired_and_unpaired_items(
        self, index: int, 
        keys: list, 
        meta: dict,
        unpaired_ma_orders: list,
        unpaired_ma_keys: list,
        unpaired_ma_meta: dict,
        unpaired_mf_orders: list,
        unpaired_mf_keys: list,
        unpaired_mf_meta: dict
    ):
        n = self.num_datasets
        sub_index = (index // n) % len(keys)
        ma_img, gt_img, li_img, mask = self.get_paired_item(sub_index, keys, meta)
        unp_ma_img, unp_ma_quality = self.get_unpaired_item(index//n, unpaired_ma_orders, unpaired_ma_keys, unpaired_ma_meta)
        unp_mf_img, unp_mf_quality = self.get_unpaired_item(index//n, unpaired_mf_orders, unpaired_mf_keys, unpaired_mf_meta)
        
        if unp_mf_img is None:
            # If we do not have unpaired data (which is almost impossible), we choose 
            # a random index in the paired dataset and retrieve the artifact-free
            # groundtruth image, assigned with a high quality score (9 ~ 10). 
            # If we do not need to use unpaired data (e.g., training a supervised MAR
            # network only requires paired data), we can just ignore the unpaired output.
            rand_index = self._get_another_index(keys, sub_index)
            _, unp_mf_img, _, _ = self.get_paired_item(rand_index, keys, meta)
            unp_mf_quality = np.random.uniform(low=9, high=10)
        
        if unp_ma_img is None:
            # [NOTE] The quality assigned in this way can be inaccurate.
            rand_index = self._get_another_index(keys, sub_index)
            _, unp_ma_img, _, _ = self.get_paired_item(rand_index, keys, meta)
            unp_ma_quality = np.random.uniform(low=1, high=6)
        
        return {
            'ma_img': ma_img,
            'gt_img': gt_img,
            'li_img': li_img,
            'mask': mask,
            'unp_ma_img': unp_ma_img,
            'unp_mf_img': unp_mf_img,
            'unp_ma_qua': unp_ma_quality,
            'unp_mf_qua': unp_mf_quality
        }
    
    @staticmethod
    def _get_another_index(lst, index):
        rand_index = np.random.choice(len(lst))
        if rand_index == index:
            rand_index = np.random.choice(len(lst))
        return rand_index
                
    @staticmethod
    def _get_file_path_from_meta(meta_data, key):
        path = meta_data[key]
        if 'root_dir' in meta_data and meta_data['root_dir'] is not None:
            path = os.path.join(meta_data['root_dir'], path)
        return path
        
    def get_paired_item(self, idx, key_list, meta_dict):
        key = key_list[idx]
        meta_data = meta_dict[key]
        gt_path = self._get_file_path_from_meta(meta_data, 'gt_img')
        gt_img = self.read_data(gt_path, clip_range=(self.min_hu, self.max_hu))
        
        ma_path = self._get_file_path_from_meta(meta_data, 'metal_img')
        ma_img = self.read_data(ma_path, clip_range=(self.min_hu, self.max_hu))
        
        li_path = self._get_file_path_from_meta(meta_data, 'li_img')
        li_img = self.read_data(li_path, clip_range=(self.min_hu, self.max_hu))
        
        mask_path = self._get_file_path_from_meta(meta_data, 'metal_mask')
        mask = self.read_data(mask_path, clip_range=(0, 1))
        
        ma_img, gt_img, li_img = self.preprocess_tensors(ma_img, gt_img, li_img)
        ma_img, gt_img, li_img, mask = self.augment_tensors(ma_img, gt_img, li_img, mask)
        return ma_img, gt_img, li_img, mask
    
    def get_unpaired_item(self, idx, orders: list, keys: list, meta: dict):
        if (len(orders) == 0) or (len(keys) == 0):
            return None, None
        key_idx = orders[idx % len(keys)]
        key = keys[key_idx]
        meta_data = meta[key]
        
        img_path = self._get_file_path_from_meta(meta_data, 'img')
        unpaired_quality = meta_data['quality']
        unpaired_img = self.read_data(img_path, clip_range=(self.min_hu, self.max_hu))
        unpaired_img = self.preprocess_tensors(unpaired_img)
        unpaired_img = self.augment_tensors(unpaired_img)
        return unpaired_img, unpaired_quality
    
    def prepare_paired_data(self, json_path):
        if json_path is not None:
            meta_dict = self.read_json_file(json_path)
            tmp = self.split_data(list(meta_dict.items()))
            key_list = list(dict(tmp).keys())
        else:
            key_list = []
            meta_dict = {}
        return key_list, meta_dict
    
    def prepare_unpaired_data(self, json_path, key='ma'):
        # `key` should be 'ma' (for metal artifact-affected images) 
        # or 'mf' (for metal artifact-free images)
        if json_path is not None:
            meta = self.read_json_file(json_path)
            assert key in meta.keys(), f"Unexpected key: {key}. The keys in `meta` are {list(meta.keys())}"
            meta = meta[key]
            tmp = self.split_data(list(meta.items()))
            keys = list(dict(tmp).keys())
            orders = list(range(len(keys)))
            np.random.shuffle(orders)
        else:
            orders = keys = []
            meta = {}
        return orders, keys, meta
        
    def read_data(self, data_path: str, clip_range: tuple, preprocess_png=True):
        if data_path.endswith('npy'):
            data = np.load(data_path)
        elif data_path.endswith('png'):
            data = np.asarray(Image.open(data_path))
            if preprocess_png:  
                # preprocessing CT HU values stored in PNG files
                data = (data - 32768).astype(np.int16)
        else:
            raise NotImplementedError(f"Unsupported data type for data path: {data_path}, should be either png or npy.")
        
        data = data.squeeze().clip(clip_range[0], clip_range[1])
        data = torch.from_numpy(data).float().unsqueeze(0)
        return data

    def split_data(self, data_list, num_train=None, num_val=None, shuffle=False):
        total_len = len(data_list)
        num_train = num_train or self.num_train
        num_val = num_val or self.num_val
        num_train = int(total_len * num_train) if (0 <= num_train <= 1) else num_train
        num_val = int(total_len * num_val) if (0 <= num_val <= 1) else num_val
        
        num_train = int(min(total_len, num_train))
        num_val = int(min(total_len, num_val))
        
        if num_train + num_val > total_len:
            warnings.warn(f'#train ({num_train}) and #test ({num_val}) are greater than #total ({total_len}).')
        
        if shuffle:
            np.random.shuffle(data_list)
            
        return data_list[:num_train] if self.mode == 'train' else data_list[-num_val:]
        
    @staticmethod
    def read_json_file(json_path):
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
        return data

    def preprocess_tensors(self, *tensors):
        tensors = [self.cttool.normalize_hu(t, do_clip=True) for t in tensors]
        return tensors[0] if len(tensors) == 1 else tensors
   
    def augment_tensors(self, *tensors):
        if self.flip_prob == 0 and self.rot_prob == 0:
            return tensors[0] if len(tensors) == 1 else tensors
        
        def maxabs(x):
            sign = 1. if x >=0 else -1.
            return max(abs(x), 5) * sign
        
        rot_angle = maxabs(np.random.uniform(-30, 30))
        if self.flip_prob > 0 and np.random.rand() < self.flip_prob:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        if self.flip_prob > 0 and np.random.rand() < (self.flip_prob ** 2):
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if self.rot_prob > 0 and np.random.rand() < self.rot_prob:
            tensors = [rotate_tensor(t, rot_angle) for t in tensors]
        
        return tensors[0] if len(tensors) == 1 else tensors
   
