import func
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def spacial_analysis_perturb(video, frame_pairs, gt_class_idx, pred_logit):

    #get gradcam mask for the input video
    GCAM_THR = 0.4
    gcam = gmodel.calc_gradcam(video.permute(1,0,2,3)[None,:].to('cuda'))
    gcam_mask = (gcam > GCAM_THR).int()
    # func.show_gray_image(gcam_mask[0,4,:].detach().cpu().numpy())

    for p in frame_pairs:
        img1 = video[p[0],:][None,:]
        img2 = video[p[1],:][None,:]
        flow = raftof.predict_flow_batch(img2, img1)
        flow = raftof.resize_flow_interpolate(flow)
        flow_mag = torch.sum(flow**2, dim=1)**0.5
        threshold = flow_mag.mean() + 0.4*flow_mag.std()
        flow_mask = (flow_mag > threshold).int()

        mask = gcam_mask[0,p[0],:] * flow_mask

        # func.show_gray_image(flow_mask[0,:].float().detach().cpu().numpy())

        # func.show_gray_image(flow_mask[0,:].float().detach().cpu().numpy())
        # func.show_rgb_image(img1[0,:].permute(1,2,0).float().detach().cpu().numpy())

        # warped = func.warp_batch(img1.float().detach(), flow.detach().cpu())
        # func.play_tensor_video_opencv(torch.stack([img1[0],warped[0,:]]),fps=1)
        # func.play_tensor_video_opencv(torch.stack([img2[0],warped[0,:]]),fps=1)
        # func.play_tensor_video_opencv(torch.stack([img1[0],img2[0,:]]),fps=1)

        ret = modify_flow(img1, img2, flow, mask)
        w = ret['warped']
        videos = torch.empty(0).to(w.device)
        for i in range(w.size(0)):
            v = video.clone().to('cuda')
            v[p[1],:] = w[i,:]
            videos = torch.cat((videos, v[None,:]), dim=0)
        with torch.no_grad():
            pred = model(videos.permute(0,2,1,3,4))
            pred = F.softmax(pred, dim=1)
            logits = pred[:,gt_class_idx]
            print(logits)

            # func.show_rgb_image(img2[0,:].permute(1,2,0).detach().cpu().numpy())



        pass
    pass


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
    FLOW_RATIO = 0.7
    window_w = 8
    MASK_THR = 0.5

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
    img_out_paht = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance_imgs'
    path = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json'
    with open(path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)

    for k in data_dict:
        d = data_dict[k]

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

            if pair_importance[0][0] == [None,None]:
                if pair_importance[0][1] < THR:
                    print('Motion is not important for this video')
                    continue

            g = k.split('_')[2][1:]
            c = k.split('_')[3][1:]
            cls_name = d['motion_importance']['gt_class']
            vid_path = ucf101dm.construct_vid_path(cls_name,g,c)
            video = ucf101dm.load_jpg_ucf101(vid_path,n=0)
            pairs = [pi[0] for pi in pair_importance if pi[0]!=[None,None]]
            frame_pairs = [(clustered_ids[str(p[0])][-1],clustered_ids[str(p[1])][0]) for p in pairs]

            ret = spacial_analysis_perturb(video, frame_pairs, gt_class_idx, pred_logit)

            for i,p in enumerate(frame_pairs):
                d = ret[i]
                dPred_dF = d['dPred_dF']
                dPred_dF_flow = d['dPred_dF*flow']
                flowmag = d['flow_mag']

                img = video[p[1],:]
                if img.size(2)!=dPred_dF.size(1):
                    img = F.interpolate(img[None,:], size=(dPred_dF.size(0), dPred_dF.size(1)), mode='bilinear', align_corners=False)
                    img=img[0,:]
                
                dPred_dF = dPred_dF.detach().cpu().numpy()
                dPred_dF = (dPred_dF-dPred_dF.min())/(dPred_dF.max()-dPred_dF.min()+1e-5)

                dPred_dF_flow = dPred_dF_flow.detach().cpu().numpy()
                dPred_dF_flow = (dPred_dF_flow-dPred_dF_flow.min())/(dPred_dF_flow.max()-dPred_dF_flow.min()+1e-5)

                flowmag = flowmag.detach().cpu().numpy()
                flowmag = (flowmag-flowmag.min())/(flowmag.max()-flowmag.min()+1e-5)


                plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
                plt.imshow(dPred_dF_flow, cmap='hot', alpha=0.5)
                # plt.imshow(mag.detach().cpu().numpy(), cmap='hot', alpha=0.5)
                # plt.imshow(slc[0,:].detach().cpu().numpy(), cmap='hot', alpha=0.5)
                plt.show(block=True)

            pass



if __name__ == '__main__':
    go_through_samples()







    





    