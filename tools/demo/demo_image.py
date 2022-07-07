# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
from PIL import Image
import cv2
import os.path as op
import argparse
import json
import sys
import os
import numpy as np
import torch
import os
import uuid
import io
# os.environ['LMDB_FORCE_CFFI'] = '1'
import lmdb
# env = lmdb.open('/home/yangdj/data/ego4d_data/det_lmdb', map_size=3099511627776)

from torch.utils.data import Dataset,DataLoader
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tqdm import tqdm
from tools.demo.detect_utils import detect_objects_on_single_image
from tools.demo.visual_utils import draw_bb, draw_rel

def cv2Img_to_Image(input_img):
    # try:
    # cv2_img = input_img.copy()
    # img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(input_img)
    return img


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]

prefix = '/home/yangdj/data/ego4d_data/'
class Ego4d(Dataset):
    def __init__(self,part,index, r=False):
        super(Ego4d).__init__()
        files = open('/home/dejie/projects/scene_graph_benchmark/all_imgs.list').readlines()
        l = len(files)
        s = 0 if index==0 else l//part*index
    
        todo_frames = [l.strip() for l in files]
        
        self.todo_list = todo_frames[s:] if index== -1 or (index+1) >=part else todo_frames[s:l//part*(index+1)]
        if r:
           self. todo_list.reverse()
        self.transforms = build_transforms(cfg, is_train=False)
    def __len__(self):
        return len(self.todo_list)
    def __getitem__(self, index):
        img_file = self.todo_list[index].strip()
        json_save_file = 'results3/'+str(uuid.uuid1()) + '.json'
        pt_save_file = json_save_file.replace('.json','.pt')
        if (os.path.exists(json_save_file) and os.path.exists(pt_save_file)):

            return None, None, None

        if not os.path.exists(img_file):
            print('error:',img_file)
            return None, None, None
        cv2_img = cv2.imread(img_file)
        cv2_img = cv2Img_to_Image(cv2_img)
        cv2_img_t, _ = self.transforms(cv2_img, target=None)
        return cv2_img_t,np.array(cv2_img.size),(json_save_file,pt_save_file)
        
def collect(batch):
    # print(batch)
    imgs = []
    sizes = []
    all_files = []
    for img,size,files in batch:
        if img is None:
            continue
        imgs.append(img)
        sizes.append(size)
        all_files.append(files)
    return imgs, sizes, all_files
def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",default='sgg_configs/vgattr/vinvl_x152c4.yaml',
                        help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument('--bz',default=6,type=int)
    parser.add_argument('--ratio')
    parser.add_argument('--reverse',action='store_true')
    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    
    model.to(cfg.MODEL.DEVICE)
    # model = model.half()
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir='')
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    # visual_labelmap is used to select classes for visualization
    # try:
    #     visual_labelmap = load_labelmap_file(args.labelmap_file)
    # except:
    #     visual_labelmap = None

    # if cfg.MODEL.ATTRIBUTE_ON:
    #     dataset_attr_labelmap = {
    #         int(val): key for key, val in
    #         dataset_allmap['attribute_to_idx'].items()}
    
    
    
    part, index = [int(r) for r in args.ratio.split(':')]
    
    # in_dir = '/home/yangdj/projects/notebooks/'
    # out_dir = '/home/yangdj/data/ego4d_data/vinvl'
    # files = open('/home/yangdj/frames_fps6.list').readlines()
    # l = len(files)
    # s = 0 if index==0 else l//part*index
    
    # todo_frames = [l.strip() for l in files]
    # todo_list = todo_frames[s:] if index== -1 or (index+1) >=part else todo_frames[s:l//part*(index+1)]
    r = False
    if args.reverse:
        r=True
    dest = Ego4d(part,index)
    
    bz = args.bz
    # dataloader = DataLoader(dest, batch_size=bz, num_workers=8)

    loader = DataLoader(dest, batch_size=bz, num_workers=8, collate_fn=collect)
    for cv2_imgs, sizes,real_files in tqdm(loader):
        # print(cv2_imgs.shape)
        # print(sizes)
        # print(real_files)
        # print(len(real_files))
        
        if len(real_files)==0:
            continue
        # imgs =[]
        results = detect_objects_on_single_image(model, cv2_imgs,sizes)
        # box_features, dets
        # print(len(results))
        assert len(real_files)==len(results)
        for (json_save_file, pt_save_file),(box_features, dets) in zip(real_files,results):
            # for obj in dets:
            #     obj["class"] = dataset_labelmap[obj["class"]]
            result = {}
            result['labels'] = [d["class"]for d in dets]
            result['rects']= [d["rect"] for d in dets]
            result['scores'] = [d["conf"] for d in dets]
            result['scores_all'] = [d["score_all"] for d in dets]
            result['attr_labels'] = [d["attr"] for d in dets]
            result['attr_scores'] = [d["attr_conf"] for d in dets]
            # with open(json_save_file, "w") as fid:
            #     json.dump(result,fid)
            # torch.save(box_features,pt_save_file)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=4 python tools/demo/demo_image.py --ratio 10:4 --bz 10