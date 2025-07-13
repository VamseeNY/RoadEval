import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageEnhance
import numpy as np
import os
import math
import segmentation_models_pytorch as smp

device = 'cpu'

# Initialize and load the trained model
model = smp.MAnet(
    encoder_name="timm-resnest26d",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)
model.load_state_dict(torch.load("road_segmentation-bm.pth", map_location=device))
model.to(device)
model.eval()

# Define the transformation for each tile.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def sliding_window_inference(image, model, transform, device, tile_size=512, stride=256):
    """
    Perform inference using a sliding window approach with overlapping tiles.
    The overlapping areas are averaged to produce a smoother segmentation.
    """
    width, height = image.size
    # Initialize matrices for accumulating predictions and counting contributions
    full_prob = np.zeros((height, width), dtype=np.float32)
    count_mat = np.zeros((height, width), dtype=np.float32)
    
    # Loop over the image with a sliding window
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            
            # Crop the tile from the original image
            tile = image.crop((left, top, right, bottom))
            tile_w, tile_h = tile.size
            
            # If the tile is smaller than tile_size, pad it to the required size
            if tile_w != tile_size or tile_h != tile_size:
                padded_tile = Image.new("RGB", (tile_size, tile_size))
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile
            
            # Preprocess the tile
            input_tensor = transform(tile).unsqueeze(0).to(device)
            
            # Run inference on the tile
            with torch.no_grad():
                output = model(input_tensor).squeeze(1)  # shape: (B, H, W)
                prob_tile = torch.sigmoid(output).cpu().numpy()[0]
            
            # If padding was added, remove it from the prediction
            prob_tile = prob_tile[:tile_h, :tile_w]
            
            # Accumulate probabilities and count the contributions
            full_prob[top:bottom, left:right] += prob_tile
            count_mat[top:bottom, left:right] += 1

    # Average the probabilities in overlapping areas
    full_prob /= count_mat
    # Convert the averaged probabilities to a binary mask using a threshold
    predicted_mask = (full_prob > 0.25).astype(np.uint8)
    return predicted_mask

# ----------------------------------------------------------------------
# Instead of scanning a folder, directly provide the image path.
image_path = r"C:\Users\vamsv\Downloads\map_snapshot2.png"
original_image = Image.open(image_path).convert("RGB")

# Optional preprocessing for satellite imagery (e.g., contrast enhancement)
enhancer = ImageEnhance.Contrast(original_image)
preprocessed_image = enhancer.enhance(1.5)

# Run sliding window inference on the preprocessed image
predicted_mask = sliding_window_inference(preprocessed_image, model, transform, device, tile_size=512, stride=256)

# -----------------------------------------------
# Geographic information and user path details.
# Assume the input image has a resolution of 1280x644 (width x height)
# and we are given:
#   - bottom_left: (lat_min, lon_min)
#   - top_right: (lat_max, lon_max)
bottom_left_geo = (22.845368, 86.195362)   # (latitude, longitude) of bottom-left
top_right_geo   = (22.849597, 86.202092)  # (latitude, longitude) of top-right

# Example list of user path points (latitude, longitude)
user_path_geo = [
    (22.845794, 86.197915),
    (22.845856, 86.198018),
    (22.845893, 86.198059),
    (22.845951, 86.198121),
    (22.846036, 86.198173),
    (22.846142, 86.198229),
    (22.846218, 86.198290),
    (22.846327, 86.198373),
    (22.846434, 86.198438),
    (22.846524, 86.198490),
    (22.846804, 86.198537),
    (22.846931, 86.198542),
    (22.847032, 86.198553),
    (22.847127, 86.198576),
    (22.847247, 86.198587),
    (22.847375, 86.198626),
    (22.847488, 86.198669),
]

def geo_to_pixel(lat, lon, bottom_left, top_right, image_width, image_height):
    """
    Convert geographic coordinates (lat, lon) to pixel coordinates.
    Assumes a linear mapping between geo bounds and image dimensions.
    """
    lat_min, lon_min = bottom_left
    lat_max, lon_max = top_right
    
    # x coordinate: based on longitude
    x = (lon - lon_min) / (lon_max - lon_min) * image_width
    # y coordinate: image y where 0 is top; lat_max corresponds to y=0.
    y = (lat_max - lat) / (lat_max - lat_min) * image_height
    return int(round(x)), int(round(y))

def check_user_path_on_road(user_path, predicted_mask, bottom_left, top_right, margin=20, match_threshold=0.50):
    """
    For each point in user_path (a list of (lat, lon) points), convert to pixel coordinates,
    then check if the surrounding region (with a given margin) contains road pixels.
    If the fraction of points that hit a road is above match_threshold, return True.
    """
    height, width = predicted_mask.shape  # (rows, cols)
    matched_points = 0
    total_points = len(user_path)
    
    for lat, lon in user_path:
        x, y = geo_to_pixel(lat, lon, bottom_left, top_right, width, height)
        # Define a buffer region around the point
        x_start = max(x - margin, 0)
        x_end   = min(x + margin, width - 1)
        y_start = max(y - margin, 0)
        y_end   = min(y + margin, height - 1)
        
        # Extract the region from the segmentation mask
        region = predicted_mask[y_start:y_end+1, x_start:x_end+1]
        # Check if the region contains any road pixel
        if np.sum(region) > 0:
            matched_points += 1
    
    match_ratio = matched_points / total_points
    return match_ratio >= match_threshold

# Use the function to check the user's path with a specified match threshold (e.g., 75%)
if check_user_path_on_road(user_path_geo, predicted_mask, bottom_left_geo, top_right_geo, margin=30, match_threshold=0.75):
    print("yes")
else:
    print("no")

# -----------------------------------------------
# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Show the preprocessed image.
axes[0].imshow(preprocessed_image)
axes[0].set_title("Preprocessed Satellite Image")
axes[0].axis("off")

# 2. Show the predicted segmentation mask.
axes[1].imshow(predicted_mask, cmap="gray")
axes[1].set_title("Predicted Road Segmentation")
axes[1].axis("off")

# 3. Show the preprocessed image with user path and buffer areas.
axes[2].imshow(preprocessed_image)
axes[2].set_title("User Path with Buffer Areas")
axes[2].axis("off")

# Draw user path points and their buffer areas.
for lat, lon in user_path_geo:
    x, y = geo_to_pixel(lat, lon, bottom_left_geo, top_right_geo, preprocessed_image.width, preprocessed_image.height)
    # Add a circle patch representing the buffer area.
    circle = Circle((x, y), 10, edgecolor='blue', facecolor='none', linewidth=2, alpha=0.8)
    axes[2].add_patch(circle)
    # Mark the central user point.
    axes[2].plot(x, y, 'ro')

plt.tight_layout()
plt.show()
