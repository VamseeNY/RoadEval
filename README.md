# RoadEval
- A scalable road mapping system that combines user trip data from EV bikes with satellite imagery to detect frequently traveled, unmapped roads.
- This project uses a custom road segmentaion moddel trained on the DeepGlobe Road Extraction dataset (https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)

## Pipeline
1. Data Collection and model initialization
  - Collect GPS trip data from bikes (synthetic GPS data was created due to the unavailability of actual data).  
  - Detect paths taken by users that are not mapped.
  - Fetch corresponding satellite imagery for the region
  - The segmentation model is loaded and set to eval. 
2. Road Segmentation
  - A map snapshot(image) is read in followed by preproessing (contrast enhancement, and normalization)
  - Sliding window inference inference followed by thresholding to get binary segmentation mask
3. Geographic mapping
    - Maps real-world GPS coordinates to pixel coordinates on the segmented image, using the known geographic bounds of the image (bottom-left and top-right latitude/longitude).
4. Path validation
  - For each user path point, a buffer region (30-pixel radius) is defined around the corresponding pixel.
  - The script checks if any road pixels are present within each buffer region in the segmentation mask.
  - If the fraction of user path points that overlap with detected roads exceeds a specified threshold (75%), the path is considered to follow an actual road.
4. Web Interface
  - Users can view, analyze, and validate detected roads.

## Sample Cases
### Case 1:
Unmapped Road
<img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/2ec938bb-420d-473e-bc26-ea8c96532e6a" />
Pipeline output
<img width="1722" height="319" alt="image" src="https://github.com/user-attachments/assets/a8c11dec-c83e-4c5b-b8a6-2f157ae67127" />

### Case 2:
Unmapped Road
<img width="1919" height="928" alt="image" src="https://github.com/user-attachments/assets/6986b9b3-8980-4d32-acfc-10ec7f3840cc" />
Pipeline output
<img width="1722" height="328" alt="image" src="https://github.com/user-attachments/assets/5250c48f-9704-4d2e-84c7-d3cf758ff27b" />

## Segmentation model
- A MANet was trained on the DeepGlove Road Extraction dataset over 10 epochs (minimal hyperparameter tuning and epochs were set due to time constraints at the hackathon)
### Performance metrics
<img width="1397" height="579" alt="image" src="https://github.com/user-attachments/assets/049786a3-7691-47c0-9af8-ece5c2bbf805" />


## Project structure
├── backend.py      # FastAPI server: APIs, model inference, and map HTML generation
├── frontend.py     # Async client: collects trip data, fetches map, user interaction
├── main.py         # Simple API for adding/updating places (demo purposes)
├── static/         # Static files (segmentation masks, analyzed maps)
├── templates/      # HTML templates for map visualization
├── road_segmentation-bm.pth # Trained model weights
├── case1.py # pipeline on sample 1 
├── case2.py # pipeline on sample 2
