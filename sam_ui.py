import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )




np.random.seed(3)

def show_gray_image(gray):
    import matplotlib.pyplot as plt
    gray =  (gray - gray.min())/(gray.max() - gray.min() + 1e-5)
    gray = gray * 255.0
    gray = gray.astype(np.uint8)
    plt.imshow(gray, cmap='gray')
    plt.axis("off")
    plt.show()


def show_rgb_image(img):
    import matplotlib.pyplot as plt
    img =  (img - img.min())/(img.max() - img.min() + 1e-5)
    img = img * 255.0
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def generate_distinct_colors_matplotlib(n_colors, colormap='tab20'):
    """
    Generate distinct colors using matplotlib colormaps
    
    Args:
        n_colors: Number of colors needed
        colormap: Matplotlib colormap name ('tab10', 'tab20', 'Set1', 'Set2', 'Set3')
    
    Returns:
        List of (R, G, B) tuples with values 0-255
    """
    cmap = cm.get_cmap(colormap)
    colors = []
    
    for i in range(n_colors):
        # Get color from colormap
        rgba = cmap(i % cmap.N)
        rgb = rgba[:3]  # Remove alpha channel
        
        # Convert to 0-255 range
        rgb_255 = tuple(int(c * 255) for c in rgb)
        colors.append(rgb_255)
    
    return colors

class Seg_UI:
    def __init__(self, prev_masks=None):
        self.prev_masks = prev_masks
        sam2_checkpoint = r"C:\Users\lahir\code\motion_vis\sam2\checkpoints\sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        self.positive_pts = np.empty((0, 2))
        self.positive_lbls = np.empty((0))

        self.negative_pts = np.empty((0, 2))
        self.negative_lbls = np.empty((0))

        self.original_image = None
        self.display_image = None
        self.window_name = "Select Objects"
        
        # Store the latest mask
        self.current_mask = None
        self.mask_overlay_alpha = 0.5  # Transparency for mask overlay
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        clicked = False
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True
            self.positive_pts = np.append(self.positive_pts, np.array([[x,y]]), axis=0)
            self.positive_lbls = np.append(self.positive_lbls, np.array([1]),axis=0)
            
        if event == cv2.EVENT_RBUTTONDOWN:
            clicked = True
            self.negative_pts = np.append(self.negative_pts, np.array([[x,y]]), axis=0)
            self.negative_lbls = np.append(self.negative_lbls, np.array([0]),axis=0)
        
        if clicked:
            input_point = np.append(self.positive_pts, self.negative_pts, axis=0)
            input_label = np.append(self.positive_lbls, self.negative_lbls, axis=0)

            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            m = masks>0
            if len(m) > 0:
                self.current_mask = m[0]
                self.update_display(input_point, input_label)

    def overlay_mask(self, image, mask, color=(0, 255, 0), alpha=0.5):
        """Overlay mask on image with specified color and transparency"""
        overlay = image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        # Apply overlay
        cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
        
        # Combine with original image where mask is False
        result = np.where(mask[:, :, np.newaxis], overlay, image)
        
        return result

    def draw_points(self, image, points, labels, point_size=5):
        """Draw positive (green) and negative (red) points on image"""
        img_copy = image.copy()
        
        # Draw positive points (green)
        if len(self.positive_pts) > 0:
            for pt in self.positive_pts:
                cv2.circle(img_copy, tuple(pt.astype(int)), point_size, (0, 255, 0), -1)
                cv2.circle(img_copy, tuple(pt.astype(int)), point_size + 2, (255, 255, 255), 1)
        
        # Draw negative points (red)
        if len(self.negative_pts) > 0:
            for pt in self.negative_pts:
                cv2.circle(img_copy, tuple(pt.astype(int)), point_size, (0, 0, 255), -1)
                cv2.circle(img_copy, tuple(pt.astype(int)), point_size + 2, (255, 255, 255), 1)
        
        return img_copy

    def update_display(self, input_point=None, input_label=None):
        """Update the displayed image with current mask and points"""
        if self.original_image is None:
            return
            
        # Start with original image
        display = self.original_image.copy()
        
        # Add mask overlay if available
        if self.current_mask is not None:
            display = self.overlay_mask(display, self.current_mask, 
                                       color=(0, 255, 0), alpha=self.mask_overlay_alpha)
        
        # Draw points
        display = self.draw_points(display, self.positive_pts, self.negative_pts)
        
        # Add info text
        info_text = f"Pos: {len(self.positive_pts)} | Neg: {len(self.negative_pts)}"
        cv2.putText(display, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if input_point is not None and len(input_point) > 0:
            coord_text = f"Last point: ({input_point[-1][0]}, {input_point[-1][1]})"
            cv2.putText(display, coord_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update the display
        self.display_image = display
        cv2.imshow(self.window_name, display)

    def add_masks(self, image, masks):
        n_col = len(masks)+1
        colors = generate_distinct_colors_matplotlib(n_col)

        img = image.copy()
        for i in range(len(masks)):
            img = self.overlay_mask(img, masks[i], color=colors[i])
        return img
        
        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

    def ui_image(self, img_path):
        image = cv2.imread(img_path)
        if self.prev_masks is not None:
            image = self.add_masks(image, self.prev_masks)
        self.original_image = image
        self.predictor.set_image(image)
        window_name = "Select Objects"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        cv2.imshow(window_name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        while True:
            # Wait for key press (1 ms delay)
            key = cv2.waitKey(1) & 0xFF
            
            # Check for specific keys
            if key == ord('q'):  # Press 'q' to quit
                print("Quitting...")
                break
            elif key == ord('s'):  # Press 's' to save
                self.done = True
                break

        cv2.destroyAllWindows()




img_path = r'C:\Users\lahir\code\motion_vis\sam2\notebooks\images\truck.jpg'

prev_masks = []
while True:
    if len(prev_masks) == 0:
        segui = Seg_UI()
    else:
        segui = Seg_UI(prev_masks=prev_masks)
    segui.ui_image(img_path)
    if type(segui.current_mask)==np.ndarray:
        prev_masks.append(segui.current_mask)
    if segui.done:
        break
pass


