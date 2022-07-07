import torch
import json
import lmdb
import tqdm
import pickle


# env = lmdb.open('slowfast_train_lmdb', map_size=1099511627776)


# with open('/home/dejie/projects/TestDemo/data/train_info_v5.json', 'r') as f:
#     train_info = json.load(f)
# i=0
# c = 0
# txn = env.begin()
# for info in tqdm.tqdm(train_info):
    # txn = env.begin()
    # vid = info['video_uid']
    # ps=  info['p_start']
    # pe = info['p_end']
    # all = torch.load('/data0/shared/ego4d_data/v1/slowfast8x8_r101_k400/'+vid+'.pt')
    # index = list(range(ps//16,pe//16))
    # a = txn.get(str(i).encode())
    # print(pickle.loads(a).shape)
    # c+=len(index)
    # while index[-1]>len(all)-1:
    #     index = index[:-1]
    # data = all[index]
    # txn.put(str(i).encode(),pickle.dumps(data))
    # txn.commit()
    # i+=1


# print(c)


with open('/home/dejie/projects/TestDemo/data/val_info_v5.json', 'r') as f:
    val_info = json.load(f)
i=0
c = 0
env = lmdb.open('slowfast_val_lmdb', map_size=1099511627776)
txn = env.begin()
for item in tqdm.tqdm(val_info):
#     # txn = env.begin(write=True)
#     keys = list(item['clip_info'].keys())
#     ps = 999999999999
#     pe = -1
#     for k in keys:
#         ts = item['clip_info'][k]['parent_start_frame']
#         te = item['clip_info'][k]['parent_end_frame']
#         ps = min(ts,ps)
#         pe = max(te,pe)
#     # vid = item['video_uid']
#     # all = torch.load('/data0/shared/ego4d_data/v1/slowfast8x8_r101_k400/'+vid+'.pt')
#     index = list(range(ps//16,pe//16))
#     if len(index)>20:
#         print(item)
#     c+=len(index)
#     # data = all[index]
#     # if data.numpy().shape[-1]!=2304:
#     #     print(data.numpy().shape)
#     # txn.put(str(i).encode(),pickle.dumps(data))
#     # txn.commit()
    a = txn.get(str(i).encode())
    print(pickle.loads(a).shape)
    i+=1

# print(c)