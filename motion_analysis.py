import json
from models.resnet3d.main import generate_model, get_inference_utils, resume_model
from models.resnet3d.model import generate_model, make_data_parallel
from pathlib import Path
import torch
from glob import glob
import numpy as np
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import func
import json

def read_json_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
    else:
        data_dict = {}
    return data_dict

def read_json_line(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

'''
replace the whole video with just one frame repeated
and see how the prediction changes
'''
#**************************************************************************************
def motion_importance(model, video, class_names):
    # video = ucf101dm.load_jpg_ucf101(vid_path)
    # video = video.unsqueeze(0).permute(0,2,1,3,4)
    pred = model(video)
    pred = F.softmax(pred,dim=1)
    pred_cls = torch.argmax(pred,dim=1).item()

    logit_original = pred[:,pred_cls].item()
    logits = []
    preds = []
    for idx in range(video.size(2)):
        frame = video[:,:,idx,:].unsqueeze(2).repeat(1,1,16,1,1)
        pred_ = model(frame)
        pred_ = F.softmax(pred_,dim=1)
        pred_cls_ = torch.argmax(pred_,dim=1)
        preds.append(pred_cls_.item())
        logits.append(pred_[:,pred_cls].item())

    #is there at least one frame that correctly predicts the original class?
    any_correct = any([p==pred_cls for p in preds])

    max_logit = max(logits)
    lowest_perc_change = (logit_original - max_logit)/logit_original

    ret = {
        'pred_original_class': class_names[pred_cls],
        'pred_original_idx': pred_cls,
        'pred_original_logit': logit_original,
        'all_logits': logits,
        'max_frame_logit': max_logit,
        'lowest_perc_change': lowest_perc_change,
        'any_frame_correct': any_correct
    }

    return ret

MAX_VID = 10

def replace_cluster(clustered_ids, source_key, dest_key):
    ids = clustered_ids.copy()
    l = len(clustered_ids[dest_key]) + len(clustered_ids[source_key])
    ids[dest_key] = [clustered_ids[source_key][0]]*l
    del ids[source_key]
    # ordered_keys = dict(sorted(ids.items(), key=lambda x: x[1][0]))
    new_key = clustered_ids[source_key][0]
    if new_key!=dest_key:
        ids[new_key] = ids.pop(dest_key)
    ids = dict(sorted(ids.items(), key=lambda x:x[1][0]))
    return ids

def get_uniqueval_indices(ids):
    unique_ids = np.unique(list(ids.keys()))
    args = [np.argwhere(ids==id) for id in unique_ids]
    cluster_ids = {}
    for i, id in enumerate(unique_ids):
        cluster_ids[id.item()] = [int(i) for i in args[i][:,0]]
    return cluster_ids


change_threshold = 0.02
def calc_video_motion_importance(model, video, gt_class_name, gt_idx, class_names, max_solutions=100):
    file_analysis = {}

    ret = motion_importance(model, video.unsqueeze(0), class_names)
    pred_logit = ret['pred_original_logit']
    pred_class = ret['pred_original_class']
    pred_class_idx = ret['pred_original_idx']

    if pred_class_idx != gt_idx: # we conly consider correctly classified samples
        return -1

    all_logits = ret['all_logits']
    lowest_perc_change = ret['lowest_perc_change']
    
    ret['gt_class'] = gt_class_name 

    file_analysis['motion_importance'] = ret

    if lowest_perc_change > change_threshold: #there is no single frame that can exaplain the whole video
        found_sol=False
        file_analysis['single_frame_structure'] = False
        sorted_indices = [i for i, _ in sorted(enumerate(all_logits), key=lambda x: x[1], reverse=True)]
        file_analysis['sorted_importance_frame_idx'] = sorted_indices

        # prev_logit = 0
        used_values = [sorted_indices[0]]                
        for i in range(1, len(sorted_indices)):
            new_idx = sorted_indices[i]
            clusters = func.temporal_freeze(used_values + [new_idx])
            clustered_ids = clusters['cluster_ids']
            ordered_keys = clusters['ordered_keys']

            #update the video with only the valuable frames
            video_ = func.create_new_video(video, clustered_ids, ordered_keys)
            ret = func.get_pred_stats(model, video_, gt_idx, pred_logit)

            used_values.append(new_idx)
            percent_change_ = (pred_logit - ret['logit'])/pred_logit

            if percent_change_ <= change_threshold: #the video can be explained well (enough) just with the given set of frames
                found_sol = True
                pair_analysis = {}
                pair_analysis['all_imp_pairs_logit'] = ret['logit']
                pair_analysis['all_imp_pairs_per_change'] = percent_change_

                #check if the motion among the frames are important for this prediction
                clustered_ids = dict(sorted(clustered_ids.items(), key=lambda x:x[1][0]))
                pairs = func.get_motion_pairs(clustered_ids)

                # check if there are adjecent frames with insignificant motion
                # if there are any get rid of these pairs

                # p = pairs[0]
                # img1 = video[:,3,:].permute(1,2,0)
                # img2 = video[:,12,:].permute(1,2,0)
                # v = torch.cat((img1[None,:],img2[None,:]), dim=0)
                # func.play_tensor_video_opencv(v,fps=1)

                # func.play_tensor_video_opencv(v_.permute(1,2,3,0),fps=2)
                
                replace_order = []
                p = pairs[0]
                used_pairs = []
                while(p != -1):
                    c0 = replace_cluster(clustered_ids, p[0], p[1])
                    v0 = func.create_new_video(video, c0)
                    c1 = replace_cluster(clustered_ids, p[1], p[0])
                    v1 = func.create_new_video(video, c1)

                    r0 = func.get_pred_stats(model, v0, gt_idx, ret['pred_logit'])
                    r1 = func.get_pred_stats(model, v1, gt_idx, ret['pred_logit'])

                    maxlog = max(r0['logit'],r1['logit'])
                    pc = (ret['pred_logit'] - maxlog)/ret['pred_logit']
                    if pc <= 0.005:
                        d_ = {}
                        d_['current_cluster_id'] = clustered_ids
                        if r0['logit']<r1['logit']:
                            clustered_ids = c1
                            replacement = (p[1], p[0])
                        else:
                            clustered_ids = c0
                            replacement = (p[0], p[1])
                        d_['replacement'] = replacement
                        replace_order.append(d_)

                    used_pairs.append(p)
                    pairs = func.get_motion_pairs(clustered_ids)
                    p = next(filter(lambda p: p not in used_pairs, pairs), -1)

                clustered_ids = dict(sorted(clustered_ids.items(), key=lambda x:x[1][0]))
                v_= func.create_new_video(video, clustered_ids)
                r_ = func.get_pred_stats(model, v_, gt_idx, pred_logit)  
                pair_analysis['selected_pairs_logit'] = r_['logit']
                pair_analysis['replace_order'] = replace_order

                pairs = func.get_motion_pairs(clustered_ids)

                pair_avg_logit = []
                if len(pairs)==2:
                    pair = pairs[1]
                    v_ = func.create_new_video(video, clustered_ids, (pair[1], pair[0]))
                    r_ = func.get_pred_stats(model, v_, gt_idx, pred_logit)
                    pair_avg_logit.append((pair, r_['per_change']))
                else:
                    pairs.insert(0,[])
                    frames = list(clustered_ids.keys())
                    for pair in pairs:
                        existing = pair
                        forbidden = [p for p in pairs[1:] if p != pair]
                        numbers = [f for f in frames if f not in existing]
                        existing = [list(e) for e in [existing]]
                        solutions = func.sample_fill_array(numbers, existing, forbidden, max_solutions=max_solutions)
                        avg_pred = func.get_avg_pred(model, video, clustered_ids, solutions)
                        avg_logit = avg_pred[:,gt_idx].item()
                        pair_avg_logit.append((pair, avg_logit))

                pair_analysis['pair_importance'] = pair_avg_logit
                pair_analysis['clustered_ids'] = clustered_ids
                file_analysis['pair_analysis'] = pair_analysis

                #break if the first solution is found s.t. percent_change_ <= change_threshold
                break
        assert not ((gt_class_name.lower()==pred_class.lower()) and not found_sol), "Error. The prediction is correct but no solution found"
    else: # there is atleas a single frame that can explain the video classificaiton
        file_analysis['single_frame_structure'] = True

    return file_analysis


'''
UCF101 accuracy = 0.854
'''
import os

def motion_importance_UCF101():
    #create model and data loader
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    model.to('cuda')
    model.eval()
    inference_loader = ucf101dm.inference_loader
    class_names = ucf101dm.inference_class_names

    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k

    output_path = Path(r'C:\Users\lahir\Downloads\UCF101\analysis\UCF101_motion_importance.json')
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        data_dict = read_json_file(output_path)
        if len(data_dict) == 0:
            n_samples = 0
            n_correct = 0
        else:
            n_samples = data_dict['n_samples']
            n_correct = data_dict['n_correct']
        if n_samples>0:
            print(f'{n_samples/len(inference_loader)*100:.0f} % is done. ({n_samples} of {len(inference_loader)}). acc = {(n_correct/n_samples)*100:.2f}%', end='\r')

        inputs, targets = batch
        video = inputs[0].to('cuda')
        gt_class_name = targets[0][0].split('_')[1]
        gt_idx = class_labels[gt_class_name.lower()]
        file_name = targets[0][0]

        if file_name in data_dict: continue

        file_analysis = calc_video_motion_importance(model, video, gt_class_name, gt_idx, class_names)

        if file_analysis!=-1:
            n_correct += 1
            seq = targets[0][1]
            seq = ','.join([str(s) for s in seq])
            file_analysis['seq'] = seq
        n_samples += 1

        data_dict[file_name] = file_analysis
        data_dict['n_samples'] = n_samples
        data_dict['n_correct'] = n_correct

        with open(output_path, "w") as f:
            json.dump(data_dict, f)


def motion_importance_ssv2():
    from models.ssv2 import VJEPA2
    from dataloaders import ssv2

    model = VJEPA2()
    model.model.eval()
    class_names = list(model.label2id.keys())
    output_path = Path(r'C:\Users\lahir\Downloads\UCF101\analysis\ssv2_motion_importance.json')
    if os.path.exists(output_path):
        data = read_json_line(output_path)
        n_samples = len(data)
    else:
        n_samples = 0

    d_names, paths = ssv2.get_ssv2_paths()
    for d,p in zip(d_names, paths):
        file_name = p.name.split('.')[0]
        key = f'{d}_{file_name}'

        print(f'{n_samples/len(d_names)*100:.0f} % is done. ({n_samples} of {len(d_names)})', end='\r')
        n_samples += 1

        gt_idx = model.label2id[d]
        v = model.video_from_path(p)['pixel_values'][0,:].permute(1,0,2,3)
        file_analysis = calc_video_motion_importance(model, v, d, gt_idx, class_names, max_solutions=2)
        if file_analysis == -1:
            continue
        
        d_ = {key: file_analysis}
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(d_, f)
            f.write("\n")

#**************************************************************************************

def replace_frames(video, source_key, dest_key, frame_cluster_idxs):
    new_video = video.clone()
    source_frame = new_video[:,frame_cluster_idxs[source_key][0],:]
    dest_idxs = frame_cluster_idxs[dest_key]
    new_video[:,dest_idxs,:] = source_frame.unsqueeze(1).repeat(1,len(dest_idxs),1,1)
    return new_video

            
def print_clus_ids(c):
    for k in c.keys():
        print(f'{k} : {c[k]}')


if __name__ == '__main__':
    motion_importance_ssv2()






