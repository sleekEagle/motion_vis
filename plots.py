import func
import glob 
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import func

def plot_intro_image():
    json_path = r'C:\Users\lahir\Downloads\UCF101\analysis\UCF101_motion_importance.json'
    result_path = r'C:\Users\lahir\Downloads\UCF101\analysis\ucf101_spacial\Archery\v_Archery_g05_c03'

    def read_image_to_array(image_path):
        """Read image file to numpy array using PIL"""
        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array

    def add_padding_pil(image_path, padding_width, padding_color=(255, 255, 255)):
        """
        Add padding to the right side of an image using PIL
        
        Args:
            image_path: Path to input image
            padding_width: Number of pixels to add to the right
            padding_color: RGB tuple for padding color (default: black)
        """
        # Open image
        img = Image.open(image_path)
        
        # Get original dimensions
        width, height = img.size
        
        # Create new image with padding
        new_width = width + padding_width
        new_img = Image.new(img.mode, (new_width, height), padding_color)
        
        # Paste original image onto the new image (left-aligned)
        new_img.paste(img, (0, 0))
        img_array = np.array(new_img)
        
        return img_array

    data = func.read_json_line(json_path)
    idx = [i for i,d in enumerate(data) if list(d.keys())[0]=='v_Archery_g05_c03'][0]
    pi = data[idx]['v_Archery_g05_c03']['pair_analysis']['pair_importance'][1:]
    imp_vals = [i[1] for i in pi]

    #load images
    pad = 10
    path = os.path.join(result_path, f"img_*")
    files = glob.glob(path)
    imgs = [add_padding_pil(file, padding_width=pad) for file in files]
    img_stack = np.hstack(imgs)
    img_stack = img_stack[:,0:-pad,:]

    #plot the importance at the in-betweens among the images
    x_val = 0
    w = imgs[0].shape[0]
    x_val_ar = []
    for i in range(len(imgs)-1):
        if i==0: x_val = w + pad/2
        else:
            x_val += w + pad
        x_val_ar.append(x_val)

    #load overlay images
    path = os.path.join(result_path, f"overlay_*")
    files = glob.glob(path)
    imgs = [add_padding_pil(file, padding_width=pad) for file in files]
    overlay_stack = np.hstack(imgs)
    overlay_stack = overlay_stack[:,0:-pad,:]

    p = img_stack.shape[1] - overlay_stack.shape[1]
    pad = np.ones((img_stack.shape[0], p, 3))*255
    overlay_stack = np.concatenate([pad, overlay_stack],axis=1)
    overlay_stack = overlay_stack.astype(np.uint8)

    # Create figure
    plt.rcParams.update({
        'font.family': 'serif',           # Font family: serif, sans-serif, monospace
        'font.size': 9,                   # Base font size
        'font.weight': 'normal',           # normal, bold
        'font.style': 'normal',            # normal, italic, oblique
    })

    fig, ax = plt.subplots(figsize=(12, 50))
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    # Display image
    im1 = ax.imshow(img_stack)
    ax.set_title('(a) Original Images', y=-0.15)

    # Create a divider for the axes
    divider = make_axes_locatable(ax)

    # Create an axis below the image for the values
    ax_values = divider.append_axes("bottom", size="30%", pad=0.25)
    ax_img2 = divider.append_axes("bottom", size="100%", pad=0.4)

    # Plot values as points at the bottom
    # ax_values.plot(x_val_ar, imp_vals, 'ro-', markersize=8)
    bars = ax_values.bar(x_val_ar, imp_vals, width=5, color='red', alpha=0.7)
    ax_values.set_ylim(min(imp_vals)-0.05, min(1.0, max(imp_vals)+0.05))
    ax_values.set_xlim(0, img_stack.shape[1])
    ax_values.set_title('(b) Importance Score', y=-0.55)
    ax_values.set_xticks([])  # Remove x-axis ticks

    im2 = ax_img2.imshow(overlay_stack, aspect='auto')  # Your second image array
    # ax_img2.set_ylabel('Motion Saliency')
    ax_img2.set_xlim(0, overlay_stack.shape[1])
    ax_img2.set_ylim(overlay_stack.shape[0], 0)
    ax_img2.set_xticks([])  # Remove x-axis ticks
    ax_img2.set_yticks([])  # Remove y-axis ticks
    ax_img2.set_title('(c) Motion Saliency Maps', y=-0.15)


    # Remove spines from first image
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove spines from bar plot
    ax_values.spines['top'].set_visible(False)
    ax_values.spines['right'].set_visible(False)
    # Keep bottom and left spines for bar plot if you want axes

    # Remove spines from second image
    ax_img2.spines['top'].set_visible(False)
    ax_img2.spines['bottom'].set_visible(False)
    ax_img2.spines['left'].set_visible(False)
    ax_img2.spines['right'].set_visible(False)

    plt.savefig('plots/into_img.png', dpi=600, bbox_inches='tight')

'''
UCF101
*************************************************************************
avg acc: 0.8541
avg n frames: 1.9741
avg per change: 0.0702
*************************************************************************
'''
def basic_stats():
    # ucf101dm = func.UCF101_data_model()
    # inference_loader = ucf101dm.inference_loader
    # n_samples = len(inference_loader)
    n_samples = 3783

    output_path = Path(r'C:\Users\lahir\Downloads\UCF101\analysis\UCF101_motion_importance.json')
    if os.path.exists(output_path):
        data = func.read_json_line(output_path)
    acc = len(data)/n_samples
    avg_per_change = 0
    avg_n_frames = 0
    for d in data:
        dict = d[list(d.keys())[0]]
        if dict['single_frame_structure']:
            avg_n_frames += 1
            no_motion_logit = dict['motion_importance']['max_frame_logit']
            orig_logit = dict['motion_importance']['pred_original_logit']
            avg_per_change += (orig_logit-no_motion_logit)/orig_logit
            continue
        else:
            avg_n_frames += len(list(dict['pair_analysis']['clustered_ids'].keys()))
            pi = dict['pair_analysis']['pair_importance']
            assert pi[0][0] == [] or len(pi) == 1, 'Error in pair importance'
            if pi[0][0] == []:
                no_motion_logit = pi[0][1]
            elif len(pi) == 1:
                no_motion_logit = pi[0][1]
            orig_logit = dict['motion_importance']['pred_original_logit']
            avg_per_change += (orig_logit-no_motion_logit)/orig_logit
    
    avg_per_change/=n_samples
    avg_n_frames/=n_samples

    print('*************************************************************************')
    print(f'avg acc: {acc:.4f}')
    print(f'avg n frames: {avg_n_frames:.4f}')
    print(f'avg per change: {avg_per_change:.4f}')
    print('*************************************************************************')
    

if __name__ == '__main__':
    basic_stats()












