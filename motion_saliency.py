import func
import json
import torch


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

def spacial_analysis(video, frame_pairs):
    gmodel.zero_grad()
    input = video.permute(1,0,2,3)[None,:]
    input.requires_grad = True
    input = input.to('cuda')
    input.retain_grad()  
    pred = gmodel(input)
    pred_idx = torch.argmax(pred,dim=1)
    pred[0,pred_idx].backward()
    gcam = gmodel.calc_gradcam(input)
    gmodel.calc_flow_saliency(input, frame_pairs)

    pass


def go_through_samples():
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
            spacial_analysis(video, frame_pairs)



if __name__ == '__main__':
    go_through_samples()







    





    