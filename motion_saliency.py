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
THR = 0.05

gmodel = func.GradcamModel(model)
gmodel.to('cuda')

raftof = func.RAFT_OF()

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

def spacial_analysis_perturb(video, frame_pairs):
    for p in frame_pairs:
        img1 = video[p[0],:][None,:]
        img2 = video[p[1],:][None,:]
        flow = raftof.predict_flow_batch(img1, img2)
        modify_flow(img1[0,:],img2[0,:],flow[0,:])
    pass


def get_windows(t, window_w, stride):
    B,C,H,W = t.size()
    windows = F.unfold(t, kernel_size=(window_w, window_w), stride=stride)
    windows = windows.view(B, C, window_w, window_w, int(H/window_w), int(H/window_w))
    return windows


def modify_flow(img1, img2, flow):
    FLOW_RATIO = 0.9
    window_w = 8

    if img1.size(2)!=flow.size(2):
        img1 = F.interpolate(img1[None,:], size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)[0,:]
        img2 = F.interpolate(img2[None,:], size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)[0,:]
    
    C,H,W = img1.size()
    n_tiles_w = H//window_w
    img1_w = img1[:,:,:,None,None].repeat(1,1,1,n_tiles_w,n_tiles_w).to(flow.device)
    img1_w = img1_w.view(3,H,W,-1).permute(3,0,1,2)
    img2_w = img2[:,:,:,None,None].repeat(1,1,1,n_tiles_w,n_tiles_w).to(flow.device)
    img2_w = img2_w.view(3,H,W,-1).permute(3,0,1,2)

    flow_w = flow[:,:,:,None,None].repeat(1,1,1,n_tiles_w,n_tiles_w).to(flow.device)

    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=img1.device),
        torch.arange(0, W, device=img1.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).to(flow.device)
    flow_grid = grid + flow
    mod_flow_grid = grid + flow*FLOW_RATIO

    flow_grid_w = get_windows(flow_grid[None,:].float(), window_w, window_w).long()[0,:]
    flow_grid_w = flow_grid_w.to(flow.device)
    flow_grid_w = flow_grid_w.view(2,window_w,window_w,-1)
    flow_grid_w = flow_grid_w.view(2,window_w*window_w,-1).permute(0,2,1)

    h_idx = flow_grid_w[0,:].flatten()
    w_idx = flow_grid_w[1,:].flatten()

    batch_indices = torch.arange(flow_grid_w.size(1)).repeat_interleave(flow_grid_w.size(2))

    
    # remove original regions
    img2_w[batch_indices,:,h_idx,w_idx] = 0

    # add new regions with modified optical flow
    












    ind = grid_w[0,:,:,:,0,0].view(2,64)
    ind = ind.to('cpu')

    img2[0,:,ind[0,:],ind[1,:]] = 0

    plt.imshow(img2[0,:].permute(1,2,0).detach().cpu().numpy())

    





    flow_w = get_windows(flow, window_w, window_w)

    flow_grid_w = grid_w + flow_w
    mod_flow_grid_w = grid_w + flow_w*FLOW_RATIO


    #remove the original regions from img2




    pass





    #remove the original regions from img2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(y1, y2, device=img1.device),
        torch.arange(x1, x2, device=img1.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).to(flow.device)
    grid_flow = flow[0,:,y1:y2,x1:x2].permute(1,2,0)

    orig_grid = grid + grid_flow
    mod_grid = grid + grid_flow*FLOW_RATIO
    
    raw_ycords = grid[:,:,0].flatten().long().to('cpu')
    raw_xcords = grid[:,:,1].flatten().long().to('cpu')
    
    orig_ycords = orig_grid[:,:,0].flatten().long().to('cpu')
    orig_xcords = orig_grid[:,:,1].flatten().long().to('cpu')

    mod_ycords = mod_grid[:,:,0].flatten().long().to('cpu')
    mod_xcords = mod_grid[:,:,1].flatten().long().to('cpu')

    img2_mod = img2.clone().to('cpu')
    img2_mod[:,:,orig_xcords,orig_ycords] = 0
    img2_mod[:,:,mod_xcords,mod_ycords] = img1[:,:,raw_xcords,raw_ycords]

    img2_orig = img2.clone().to('cpu')
    img2_orig[:,:,orig_xcords,orig_ycords] = 0
    img2_orig[:,:,orig_xcords,orig_ycords] = img1[:,:,raw_xcords,raw_ycords]


    imgs = [img2, img2_orig, img2_mod]
    imgs = [img[0,:].permute(1,2,0) for img in imgs]
    titles = [f'Image {i+1}' for i in range(3)]

    n_rows, n_cols = 1, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for idx, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')  # Turn off axes

    # Hide unused subplots
    for idx in range(len(imgs), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()



    plt.imshow(img2_new[0,:].permute(1,2,0).detach().cpu().numpy())
    plt.imshow(fmag[0,:].detach().cpu().numpy(), cmap='hot', alpha=0.5)
    plt.show(block=True)

def go_through_samples():
    img_out_paht = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance_imgs'
    path = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json'
    with open(path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)

    for k in data_dict:
        d = data_dict[k]

        #consider only samples that are correcly predicted
        gt_class = d['motion_importance']['gt_class']
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

            ret = spacial_analysis_perturb(video, frame_pairs)

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







    





    