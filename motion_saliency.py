import func
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

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
THR = 0.05

gmodel = func.GradcamModel(model)
gmodel.to('cuda')

raftof = func.RAFT_OF()

def show_images_seq(img1, img2):
    import numpy as np
    
    # Turn on interactive mode
    plt.ion()

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display first image
    ax.imshow(np.transpose(img1, (1, 2, 0)))
    ax.set_title("Image 1")
    ax.axis('off')
    plt.draw()
    plt.pause(2)  # Show for 2 seconds

    # Clear the figure and display second image
    ax.clear()
    ax.imshow(np.transpose(img2, (1, 2, 0)))
    ax.set_title("Image 2")
    ax.axis('off')
    plt.draw()
    plt.pause(2)  # Show for 2 seconds

    # Turn off interactive mode
    plt.ioff()
    plt.show()

def spacial_analysis(video, frame_pairs):
    gmodel.zero_grad()
    input = video.permute(1,0,2,3)[None,:]
    input.requires_grad = True
    input = input.to('cuda')
    input.retain_grad()  
    pred = gmodel(input)
    pred_idx = torch.argmax(pred,dim=1)
    pred[0,pred_idx].backward()
    ret = gmodel.calc_flow_saliency(input, frame_pairs, grad_method='gradcam')
    return ret

def spacial_analysis_perturb(video, gt_class_idx, pred_logit, ordered_keys, clustered_ids):

    #get gradcam mask for the input video
    GCAM_THR = 0.2
    gcam = gmodel.calc_gradcam(video[None,:].to('cuda'))
    gcam_mask = (gcam > GCAM_THR).int()
    # func.show_gray_image(gcam_mask[0,4,:].detach().cpu().numpy())
    pairs = func.get_motion_pairs(clustered_ids)

    heatmaps = []
    for p in pairs:
        out_dict = {}
        p_ind = {}
        ind = 0
        for k in clustered_ids:
            val = clustered_ids[k]
            ind_ar = []
            for i in range(len(val)):
                ind_ar.append(ind)
                ind+=1
            p_ind[k] = ind_ar

        p0 = p_ind[str(p[0])][0]
        p1 = p_ind[str(p[1])][0]

        img1 = video[:,p0,:][None,:]
        img2 = video[:,p1,:][None,:]
        flow = raftof.predict_flow_batch(img2, img1)
        flow = raftof.resize_flow_interpolate(flow)
        flow_mag = torch.sum(flow**2, dim=1)**0.5


        out_dict['flow_mag'] = flow_mag[0,:].detach().cpu().numpy()
        #check if this pair has sufficient motion

        # v = torch.cat((img1,img2), dim=0)
        # v = F.interpolate(v, size=(512,512), mode='bilinear', align_corners=False)

        # func.play_tensor_video_opencv(v.permute(0,2,3,1),fps=1)


        # threshold = flow_mag.mean()
        # flow_mask = (flow_mag > threshold).int()
        # out_dict['flow_mask'] = flow_mask[0,:].detach().cpu().numpy()
        # flow_mask = torch.ones_like(flow_mask)
        # func.show_gray_image(mask[0,:].detach().cpu().numpy())
        # func.show_gray_image(flow_mag[0,:].detach().cpu().numpy())

        # func.show_rgb_image(img1[0,:].permute(1,2,0).detach().cpu().numpy())

        # mask = gcam_mask[0,clustered_ids[str(p[0])][0],:] * flow_mask
        gc = gcam[0,clustered_ids[str(p[0])][0],:]
        out_dict['gcam'] = gc.detach().cpu().numpy()
        # out_dict['gcam_mask'] = gcam_mask[0,clustered_ids[str(p[0])][0],:].detach().cpu().numpy()
        # out_dict['mask'] = mask[0,:].detach().cpu().numpy()
        # mask_perc = torch.nonzero(mask).size(0)/(mask.size(1)*mask.size(2))
        # assert mask_perc>=0.05, "Not enough pixels display significant motion"

        mask = flow_mag[0,:]*gc
        v = mask>mask.mean()
        mask[v] = 1
        mask[~v] = 0
        out_dict['mask'] = mask.detach().cpu().numpy()

        # func.show_gray_image(mask.detach().cpu().numpy())

        # func.show_gray_image(gcam_mask[0,clustered_ids[str(p[0])][0],:].detach().cpu().numpy())
        # func.show_rgb_image(img1[0,:].permute(1,2,0).detach().cpu().numpy())
        # v = torch.cat((img1,img2), dim=0)
        # func.play_tensor_video_opencv(v,fps=1)

        ret = modify_flow(img1, img2, flow, mask)
        warped = ret['warped']
        mask = ret['mask']
        window = ret['window']

        videos = torch.empty(0).to(warped.device)
        for i in range(warped.size(0)):
            v = func.replace_frame(video, ordered_keys, clustered_ids, p[1], warped[i,:])
            videos = torch.cat((videos, v[None,:].to(warped.device)), dim=0)
        with torch.no_grad():
            pred = model(videos)
            pred = F.softmax(pred, dim=1)
            logits = pred[:,gt_class_idx]
        imp = pred_logit - logits
        imp = (imp - imp.min())/(imp.max()-imp.min()+1e-5)
        imp = imp[None,None,:].repeat(window,window,1)

        hm = torch.zeros(1, 1, img2.size(2), img2.size(3))
        hm_w = get_windows(hm, window, window)[0,0,:]
        hm_w[...,mask.to(hm_w.device)] = imp.to(hm_w.device)

        heatmap = hm_w[0,0,:]
        heatmap = F.interpolate(heatmap[None,None,:], size=(112,112), mode='bilinear', align_corners=False)
        heatmap = heatmap**1
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min()+1e-5)
        heatmap = -1*heatmap+1
        transformed = np.uint8(heatmap[0,0,:].numpy()*255)
        h_col = cv2.applyColorMap(transformed, cv2.COLORMAP_JET)

        img = np.uint8(((img2 - img2.min())/(img2.max() - img2.min()+1e-5)*255)[0,:].permute(1,2,0))

        final_img = cv2.addWeighted(img, 0.6, h_col, 0.4, 0)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        out_dict['heatmap'] = final_img
        heatmaps.append(out_dict)
        # cv2.imwrite('output_opencv.jpg', final_img)
        # func.show_rgb_image(final_img)
    return heatmaps


def get_windows(t, window_w, stride):
    B,C,H,W = t.size()
    windows = F.unfold(t, kernel_size=(window_w, window_w), stride=stride)
    windows = windows.view(B, C, window_w, window_w, int(H/window_w), int(H/window_w))
    return windows


'''
img1: 1,3,H,W
img2: 1,3,H,W
flow: 1,2,H,W
mask: H,W
'''
def modify_flow(img1, img2, flow, mask):
    FLOW_RATIO = 0.8
    window_w = 8
    MASK_THR = 0.8

    if img1.size(2)!=flow.size(2):
        img1 = F.interpolate(img1, size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)[0,:]
        img2 = F.interpolate(img2, size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)[0,:]
    if mask.size(0)!=flow.size(2):
        mask = F.interpolate(mask[None,:].float(), size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)[0,0,:].int()
    
    mask_w = get_windows(mask[None,None,:].float(), window_w, window_w).long()[0,0,:]
    mask_mean = (mask_w.sum(dim=(0,1))/(mask_w.size(0)**2))
    mask_mask = mask_mean > MASK_THR

    #create modified flow
    flow_w = get_windows(flow, window_w, window_w)[0,:]
    flow_w_mod = flow_w[...,mask_mask]*FLOW_RATIO
    valid_mask_idx = torch.nonzero(mask_mask)
    flow_batch = flow.repeat(valid_mask_idx.size(0),1,1,1)
    for i in range(valid_mask_idx.size(0)):
        idx = valid_mask_idx[i]
        f_mod_ = flow_w_mod[:,:,:,i]
        h_, w_ = idx[0]*window_w, idx[1]*window_w
        flow_batch[i,:,h_:h_+window_w,w_:w_+window_w] = f_mod_

    #warp the image with the modified flow
    warp = func.warp_batch(img1.repeat(valid_mask_idx.size(0),1,1,1).to('cuda'),flow_batch)

    ret = {
        'warped': warp,
        'window': window_w,
        'mask': mask_mask,
    }

    return ret


# func.show_gray_image(mask_mask.float().cpu().numpy())


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







    





    