
import func
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob

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

def go_through_samples():
    img_out_path = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance_imgs'
    path = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json'
    with open(path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)

    for i, k in enumerate(data_dict):
        print(f'Processing sample {i+1}/{len(data_dict)}: {k}', end='\r')

        d = data_dict[k]
        # if k!='v_ApplyEyeMakeup_g04_c03':
        #     continue
        #consider only samples that are correcly predicted
        gt_class = d['motion_importance']['gt_class']
        gt_class_idx = class_labels[gt_class.lower()]
        pred_logit = d['motion_importance']['pred_original_logit']
        pred_class = d['motion_importance']['pred_original_class']
        if gt_class.lower() != pred_class.lower():
            continue

        #is there a single frame that explains the whole video ?
        sfs = d['single_frame_structure']
        if sfs:
            print('There is a single frame that can explain the whole video well')
            continue
        else:
            # what sub set of frames can be used to explain the whole video ?
            clustered_ids = d['pair_analysis']['clustered_ids']
            pair_importance = d['pair_analysis']['pair_importance'] # this is percentage change. lower, better
            # for pi in pair_importance:
            #     print(f'{pi[0]}: {pi[1]}')  

            none_imp = None
            if len(pair_importance)==1:
                none_imp = pair_importance[0][1]
            elif pair_importance[0][0] == [None,None]:
                none_imp = pair_importance[0][1]
            assert none_imp is not None, "Error! None importance is not calculated for this sample"
            non_per_change = (pred_logit - none_imp)/pred_logit
            if non_per_change < 0.02:#motion is not important for this sample
                print('Motion is not important for this sample')
                continue

            g = k.split('_')[2][1:]
            c = k.split('_')[3][1:]
            cls_name = d['motion_importance']['gt_class']
            vid_path = ucf101dm.construct_vid_path(cls_name,g,c)

            #calculate segmentation masks
            files = glob.glob(f'{vid_path}\\*.jpg')
            files = sorted(files)
            from sam2.notebooks.sam_ui import get_masks_from_ui


            video = ucf101dm.load_jpg_ucf101(vid_path,n=0)
            pairs = [pi[0] for pi in pair_importance if pi[0]!=[None,None]]
            ordered_keys = list(dict(sorted(clustered_ids.items(), key=lambda x: x[1][0])).keys())

            video_ = func.create_new_video(video.permute(1,0,2,3), clustered_ids, ordered_keys)
            heatmaps = spacial_analysis_perturb(video_, gt_class_idx, d['pair_analysis']['all_imp_pairs_logit'], ordered_keys, clustered_ids)

            sub_dir = os.path.join(img_out_path, gt_class, k)
            os.makedirs(sub_dir, exist_ok=True)

            for i in range(len(heatmaps)):
                p = pairs[i]

                hm = heatmaps[i]['heatmap']
                out_path = os.path.join(sub_dir, f'hm_{p[0]}_{p[1]}.jpg')
                cv2.imwrite(out_path, hm)

                gcam = heatmaps[i]['gcam']
                gcam = np.uint8(gcam*255)
                out_path = os.path.join(sub_dir, f'gcam_{p[0]}_{p[1]}.jpg')
                cv2.imwrite(out_path, gcam)

                f = heatmaps[i]['flow_mag']
                f = np.uint8((f - f.min())/(f.max()-f.min()+1e-5)*255)
                out_path = os.path.join(sub_dir, f'fmag_{p[0]}_{p[1]}.jpg')
                cv2.imwrite(out_path, f)

                # fm = heatmaps[i]['flow_mask']
                # fm = np.uint8(fm*255)
                # out_path = os.path.join(sub_dir, f'fmask_{p[0]}_{p[1]}.jpg')
                # cv2.imwrite(out_path, fm)

                m = heatmaps[i]['mask']
                m = np.uint8(m*255)
                out_path = os.path.join(sub_dir, f'mask_{p[0]}_{p[1]}.jpg')
                cv2.imwrite(out_path, m)

                # gm = heatmaps[i]['gcam_mask']
                # gm = np.uint8(gm*255)
                # out_path = os.path.join(sub_dir, f'gmask_{p[0]}_{p[1]}.jpg')
                # cv2.imwrite(out_path, gm)

            
if __name__ == '__main__':
    go_through_samples()
