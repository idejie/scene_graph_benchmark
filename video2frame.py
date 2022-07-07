import os
import cv2
import numpy as np
import av
import json
import lmdb
import tqdm
import sys
from multiprocessing import Process
import time
def save_image(frame, video_d):
    frame.to_image().save(f'{video_d}/{frame.index:010d}.jpg')

def main():  
    with open(frame_json, 'r') as f:
        frame_dict = json.load(f)
    todo_list= list(frame_dict.items())
    l = len(todo_list)
    s = (l//part)*index

    if index>=part-1:
        todo_list = todo_list[s:]
    else:
        t = s+(l//part)
        todo_list = todo_list[s:t]
    
    l = len(todo_list)
    s = (l//part2)*index2

    if index2>=part2-1:
        todo_list = todo_list[s:]
    else:
        t = s+(l//part2)
        todo_list = todo_list[s:t]

    for video_path, frames in tqdm.tqdm(todo_list):
        video_path = os.path.basename(video_path).replace('full_scale','')
        if os.path.exists(prefix+'clips/'+video_path):
            v_p = prefix+'clips/'+video_path
        elif os.path.exists(prefix+'full_scale/'+video_path):
            v_p = prefix+'full_scale/'+video_path
        else:
            print(video_path,'not exisit')
            continue
        frames = sorted(frames)
        try:
            
            video_d = prefix+'frames/'+video_path.replace('.mp4','')
            if not os.path.exists(video_d):
                os.makedirs(video_d)
            last = frames[-1]
            if  os.path.exists(f'{video_d}/{last:010d}.jpg'):
                continue
            container = av.open(v_p)
            for frame in container.decode(video=0):
                if frame.index < frames[0]:
                    continue
                # print(f'filter:{frame.index}',time.time()-t)
                if frame.index in frames and (not os.path.exists(f'{video_d}/{frame.index:010d}.jpg')):

                    frame.to_image().save(f'{video_d}/{frame.index:010d}.jpg')

                    # Process(target=save_image,args=(frame,video_d,)).start()
                if frame.index == frames[-1]:
                    break
            # print('save',time.time()-t)
        except Exception as e:
            print(video_path,'load error',e)
        # break

if __name__ == '__main__':
    frame_json = '/home/dejie/projects/TestDemo/todo_frame.json'
    prefix = '/data0/shared/ego4d_data/v1/'
    subfix = '/data0/shared/ego4d_data/v1/frames/'
    index = int(sys.argv[1])
    part = 20
    part2 =10 
    index2 = int(sys.argv[2])
    main()
