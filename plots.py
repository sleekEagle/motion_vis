import func
import glob 
import os
from PIL import Image
import numpy as np

json_path = r'C:\Users\lahir\Downloads\UCF101\analysis\UCF101_motion_importance.json'
result_path = r'C:\Users\lahir\Downloads\UCF101\analysis\ucf101_spacial\Archery\v_Archery_g05_c03'

def read_image_to_array(image_path):
    """Read image file to numpy array using PIL"""
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

def add_right_padding_pil(image_path, padding_width, padding_color=(0, 0, 0)):
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
imgs = [add_right_padding_pil(file, padding_width=pad) for file in files]
img_stack = np.hstack(imgs)

#plot the importance at the in-betweens among the images
x_val = 0
w = imgs[0].shape[0]
x_val_ar = []
for i in range(len(imgs)-1):
    if i==0: x_val = w + pad/2
    else:
        x_val += w + pad
    x_val_ar.append(x_val)





func.show_rgb_image(img_stack)









