from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Simulated GPS data stream (replace with real GPS data in production)
gps_data = [[
    {"latitude": 22.846275, "longitude": 86.189584},
    {"latitude": 22.846339, "longitude": 86.189522},
    {"latitude": 22.846458, "longitude": 86.189429},
    {"latitude": 22.846537, "longitude": 86.189286},
    {"latitude": 22.846619, "longitude": 86.189268},
    {"latitude": 22.846747, "longitude": 86.189260},
    {"latitude": 22.846784, "longitude": 86.189321},
], [
    {"latitude": 22.845781, "longitude": 86.197974},
    {"latitude": 22.846043, "longitude": 86.198097},
    {"latitude": 22.846280, "longitude": 86.198178},
    {"latitude": 22.846606, "longitude": 86.198355},
    {"latitude": 22.846952, "longitude": 86.198371},
    {"latitude": 22.847259, "longitude": 86.198468},
    {"latitude": 22.847516, "longitude": 86.198650},
    {"latitude": 22.847704, "longitude": 86.198859},
    {"latitude": 22.847837, "longitude": 86.199079},
    {"latitude": 22.848154, "longitude": 86.199213},
    {"latitude": 22.848376, "longitude": 86.199278},
    {"latitude": 22.848386, "longitude": 86.199396},
    {"latitude": 22.848268, "longitude": 86.199777},
    {"latitude": 22.848228, "longitude": 86.199975},
]]

# Store the path and surface data for the HTML rendering
all_coordinates = []
unknown_surface_coordinates = []


class Coordinate(BaseModel):
    latitude: float
    longitude: float


def get_road_surface(lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way["highway"](around:50,{lat},{lon});
    out body;
    >;
    out skel qt;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    for element in data['elements']:
        if 'tags' in element:
            return element['tags'].get('surface', 'unknown')
    return 'unknown'


current_path_index = 0  # Tracks the current path being processed


@app.get("/get_coordinates")
async def get_coordinates():
    global current_path_index
    if current_path_index < len(gps_data) and gps_data[current_path_index]:
        coord = gps_data[current_path_index].pop(0)
        surface = get_road_surface(coord['latitude'], coord['longitude'])

        all_coordinates.append(
            {**coord, "surface": surface, "path_index": current_path_index})
        if surface == 'unknown':
            unknown_surface_coordinates.append(
                {**coord, "surface": surface, "path_index": current_path_index})

        return JSONResponse(content={**coord, "surface": surface, "path_index": current_path_index})
    elif current_path_index < len(gps_data) - 1:
        current_path_index += 1
        return await get_coordinates()
    else:
        return JSONResponse(content={"status": "end"})


@app.get("/", response_class=HTMLResponse)
async def get_map_page():
    html_content = generate_map_html()
    return HTMLResponse(content=html_content)

@app.get("/map", response_class=HTMLResponse)
async def get_map_page():
    html_content = generate_map_html()
    return HTMLResponse(content=html_content)

# Segmentation Model Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

segmentation_model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)
segmentation_model.load_state_dict(torch.load(
    "road_segmentation-bm.pth", map_location=device))
segmentation_model.to(device)
segmentation_model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def sliding_window_inference(image, model, transform, device, tile_size=512, stride=256):
    width, height = image.size
    full_prob = np.zeros((height, width), dtype=np.float32)
    count_mat = np.zeros((height, width), dtype=np.float32)

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)

            tile = image.crop((left, top, right, bottom))
            tile_w, tile_h = tile.size

            if tile_w != tile_size or tile_h != tile_size:
                padded_tile = Image.new("RGB", (tile_size, tile_size))
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile

            input_tensor = transform(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor).squeeze(1)
                prob_tile = torch.sigmoid(output).cpu().numpy()[0]

            prob_tile = prob_tile[:tile_h, :tile_w]
            full_prob[top:bottom, left:right] += prob_tile
            count_mat[top:bottom, left:right] += 1

    full_prob /= count_mat
    return (full_prob > 0.25).astype(np.uint8)


def geo_to_pixel(lat, lon, bottom_left, top_right, image_width, image_height):
    lat_min, lon_min = bottom_left
    lat_max, lon_max = top_right
    x = (lon - lon_min) / (lon_max - lon_min) * image_width
    y = (lat_max - lat) / (lat_max - lat_min) * image_height
    return int(round(x)), int(round(y))


def check_user_path_on_road(user_path, predicted_mask, bottom_left, top_right, margin=30, match_threshold=0.75):
    height, width = predicted_mask.shape
    matched_points = 0
    total_points = len(user_path)

    for lat, lon in user_path:
        x, y = geo_to_pixel(lat, lon, bottom_left, top_right, width, height)
        x_start = max(x - margin, 0)
        x_end = min(x + margin, width - 1)
        y_start = max(y - margin, 0)
        y_end = min(y + margin, height - 1)

        region = predicted_mask[y_start:y_end+1, x_start:x_end+1]
        if np.sum(region) > 0:
            matched_points += 1

    match_ratio = matched_points / total_points
    return match_ratio >= match_threshold


@app.post("/segment_image/")
async def segment_image(image_data: str = Form(...), bottom_left_lat: float = Form(...), bottom_left_lng: float = Form(...), top_right_lat: float = Form(...), top_right_lng: float = Form(...)):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        predicted_mask = sliding_window_inference(
            image, segmentation_model, transform, device)

        # Get user path from all_coordinates
        user_path = [(coord['latitude'], coord['longitude'])
                     for coord in all_coordinates]
        bottom_left = (bottom_left_lat, bottom_left_lng)
        top_right = (top_right_lat, top_right_lng)

        # Check if path is on road
        is_valid = check_user_path_on_road(
            user_path, predicted_mask, bottom_left, top_right)

        # Save mask image
        mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        mask_path = "static/segmented_mask.png"
        mask_image.save(mask_path)

        # Return both the file and the validation result
        return JSONResponse(content={
            "mask_url": "/static/segmented_mask.png",
            "is_valid": is_valid
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0).to(device)


def run_inference(model, image):
    with torch.no_grad():
        output = model(image).squeeze(1)
        predicted_mask = torch.sigmoid(output).cpu().numpy()
    return (predicted_mask > 0.5).astype(np.uint8)


@app.post("/process_image/")
async def process_image(image_data: str = Form(...)):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_image(image)
        mask = run_inference(segmentation_model, image_tensor)
        mask_image = Image.fromarray((mask[0] * 255).astype(np.uint8))
        mask_path = "static/analyzed_map.png"
        mask_image.save(mask_path)
        return JSONResponse(content={"message": "Analysis complete. Map saved as analyzed_map.png"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


def generate_map_html():
    maptiler_api_key = "uArsHJuTKubTu0k8Gj7O"

    html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rider Path on Unmapped Roads</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script src="https://unpkg.com/leaflet-image@0.4.0/leaflet-image.js"></script>
            <style>
                body { margin: 0; padding: 0; }
                #map { width: 100%; height: 100vh; }
                .info {
                    padding: 6px 8px;
                    font: 14px/16px Arial, Helvetica, sans-serif;
                    background: white;
                    background: rgba(255,255,255,0.8);
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    border-radius: 5px;
                }
                .legend {
                    line-height: 18px;
                    color: #555;
                }
                .legend i {
                    width: 18px;
                    height: 18px;
                    float: left;
                    margin-right: 8px;
                    opacity: 0.7;
                }
                #downloadButton {
                    position: absolute;
                    top: 150px;
                    right: 10px;
                    z-index: 1000;
                    background-color: white;
                    padding: 5px 10px;
                    border: 1px solid #ccc;
                    cursor: pointer;
                }
                #analyzeButton {
                    position: absolute;
                    top: 60px;
                    right: 10px;
                    z-index: 1000;
                    background-color: white;
                    padding: 5px 10px;
                    border: 1px solid #ccc;
                    cursor: pointer;
                }
                #pathToggle {
                    position: absolute;
                    top: 110px;
                    right: 10px;
                    z-index: 1000;
                    background-color: white;
                    padding: 5px 10px;
                    border: 1px solid #ccc;
                }
                #coordinatesPanel {
                    position: absolute;
                    bottom: 20px;
                    left: 20px;
                    z-index: 1000;
                    background-color: white;
                    background: rgba(255,255,255,0.9);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    max-width: 300px;
                }
                #coordinatesPanel h4 {
                    margin: 0 0 8px 0;
                    font-size: 14px;
                }
                #coordinatesPanel div {
                    margin-bottom: 4px;
                }
                #copyCoords {
                    margin-top: 8px;
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    padding: 4px 8px;
                    cursor: pointer;
                    font-size: 11px;
                }
                #copyCoords:hover {
                    background-color: #eee;
                }
                #validityPanel {
                    position: absolute;
                    bottom: 20px;
                    right: 20px;
                    z-index: 1000;
                    background-color: white;
                    background: rgba(255,255,255,0.9);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    max-width: 200px;
                }
                #validityPanel h4 {
                    margin: 0 0 8px 0;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <button id="downloadButton">Download Segmented Map</button>
            <button id="analyzeButton">Analyze Current View</button>
            <select id="pathToggle">
                <option value="all">All Paths</option>
    """

    for i in range(len(gps_data)):
        html += f'<option value="{i}">Path {i + 1}</option>'

    html += """
            </select>
            <div id="coordinatesPanel">
                <h4>Viewport Coordinates</h4>
                <div><b>Bottom-Left:</b> <span id="bottomLeft">-</span></div>
                <div><b>Top-Right:</b> <span id="topRight">-</span></div>
                <div><b>Zoom Level:</b> <span id="zoomLevel">-</span></div>
                <button id="copyCoords">Copy Coordinates</button>
            </div>
            <div id="validityPanel">
                <h4>Path Validity</h4>
                <div><b>Is Valid Road:</b> <span id="pathValidity">-</span></div>
            </div>
            <script>
    """

    html += f"""
            var map = L.map('map', {{ zoomControl: true }});

            var osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }});

            var satelliteLayer = L.tileLayer('https://api.maptiler.com/maps/satellite/{{z}}/{{x}}/{{y}}.jpg?key={maptiler_api_key}', {{
                tileSize: 512,
                zoomOffset: -1,
                minZoom: 1,
                maxZoom: 18,
                attribution: '<a href="https://www.maptiler.com/copyright/" target="_blank">© MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">© OpenStreetMap contributors</a>',
                crossOrigin: true
            }});

            satelliteLayer.addTo(map);

            var baseLayers = {{
                "OpenStreetMap": osmLayer,
                "Satellite": satelliteLayer
            }};
            L.control.layers(baseLayers).addTo(map);

            var allCoordinates = {all_coordinates};
            var unknownSurfaceCoordinates = {unknown_surface_coordinates};

            console.log('All Coordinates:', allCoordinates);
            console.log('Unknown Coordinates:', unknownSurfaceCoordinates);

            var pathLayers = {{}};
            var allPathPoints = [];

            allCoordinates.forEach(function(point) {{
                if (!pathLayers[point.path_index]) {{
                    pathLayers[point.path_index] = [];
                }}
                pathLayers[point.path_index].push([point.latitude, point.longitude]);
                allPathPoints.push([point.latitude, point.longitude]);
            }});

            var pathLayerGroups = {{}};
            for (var index in pathLayers) {{
                pathLayerGroups[index] = L.layerGroup();
                for (var i = 0; i < pathLayers[index].length - 1; i++) {{
                    var coord = allCoordinates.find(c => c.path_index == index && 
                        c.latitude == pathLayers[index][i][0] && 
                        c.longitude == pathLayers[index][i][1]);
                    var color = coord.surface === 'unknown' ? 'red' : 'blue';
                    var weight = color === 'red' ? 6 : 3;
                    L.polyline([pathLayers[index][i], pathLayers[index][i+1]], {{color: color, weight: weight}})
                        .addTo(pathLayerGroups[index]);
                }}
            }}

            var unknownMarkers = L.layerGroup();
            unknownSurfaceCoordinates.forEach(function(point) {{
                L.circleMarker([point.latitude, point.longitude], {{
                    color: 'red',
                    fillColor: '#f03',
                    fillOpacity: 0.5,
                    radius: 8
                }}).bindPopup('Unknown Surface').addTo(unknownMarkers);
            }});
            unknownMarkers.addTo(map);

            var legend = L.control({{ position: 'bottomright' }});
            legend.onAdd = function() {{
                var div = L.DomUtil.create('div', 'info legend');
                div.innerHTML = '<i style="background:#0000ff"></i> Known Surface<br><i style="background:#ff0000"></i> Unknown Surface';
                return div;
            }};
            legend.addTo(map);

            if (allPathPoints.length > 0) {{
                var bounds = L.latLngBounds(allPathPoints);
                map.fitBounds(bounds, {{ padding: [50, 50] }});
                for (var index in pathLayerGroups) {{
                    pathLayerGroups[index].addTo(map);
                }}
            }} else {{
                map.setView([22.846275, 86.189584], 15);
                console.log('No path points, using default view');
            }}

            document.getElementById('pathToggle').addEventListener('change', function(e) {{
                var selectedPath = e.target.value;
                for (var index in pathLayerGroups) {{
                    map.removeLayer(pathLayerGroups[index]);
                }}
                if (selectedPath === 'all') {{
                    for (var index in pathLayerGroups) {{
                        pathLayerGroups[index].addTo(map);
                    }}
                    if (allPathPoints.length > 0) {{
                        map.fitBounds(L.latLngBounds(allPathPoints), {{ padding: [50, 50] }});
                    }}
                }} else {{
                    pathLayerGroups[selectedPath].addTo(map);
                    if (pathLayers[selectedPath].length > 0) {{
                        map.fitBounds(L.latLngBounds(pathLayers[selectedPath]), {{ padding: [50, 50] }});
                    }}
                }}
            }});

            function updateCoordinatesDisplay() {{
                var bounds = map.getBounds();
                var bottomLeft = bounds.getSouthWest();
                var topRight = bounds.getNorthEast();
                var zoom = map.getZoom();
                document.getElementById('bottomLeft').textContent = bottomLeft.lat.toFixed(6) + ", " + bottomLeft.lng.toFixed(6);
                document.getElementById('topRight').textContent = topRight.lat.toFixed(6) + ", " + topRight.lng.toFixed(6);
                document.getElementById('zoomLevel').textContent = zoom;
            }}

            map.on('moveend', updateCoordinatesDisplay);
            setTimeout(function() {{
                updateCoordinatesDisplay();
                map.invalidateSize();
            }}, 100);

            document.getElementById('copyCoords').addEventListener('click', function() {{
                var bounds = map.getBounds();
                var bottomLeft = bounds.getSouthWest();
                var topRight = bounds.getNorthEast();
                var coordText = "Bottom-Left: " + bottomLeft.lat.toFixed(6) + ", " + bottomLeft.lng.toFixed(6) + "\\n" +
                               "Top-Right: " + topRight.lat.toFixed(6) + ", " + topRight.lng.toFixed(6) + "\\n" +
                               "Zoom Level: " + map.getZoom();
                navigator.clipboard.writeText(coordText)
                    .then(() => {{ alert("Coordinates copied to clipboard!"); }})
                    .catch(err => {{
                        console.error('Could not copy text: ', err);
                        var textArea = document.createElement("textarea");
                        textArea.value = coordText;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand("copy");
                        document.body.removeChild(textArea);
                        alert("Coordinates copied to clipboard!");
                    }});
            }});

            document.getElementById('downloadButton').addEventListener('click', function() {{
                leafletImage(map, function(err, canvas) {{
                    if (err) {{
                        console.error('Error capturing map:', err);
                        alert('Error capturing map: ' + err);
                        return;
                    }}
                    var img = document.createElement('img');
                    var dimensions = map.getSize();
                    img.width = dimensions.x;
                    img.height = dimensions.y;
                    img.src = canvas.toDataURL();

                    var bounds = map.getBounds();
                    var bottomLeft = bounds.getSouthWest();
                    var topRight = bounds.getNorthEast();

                    fetch('/segment_image/', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                        body: 'image_data=' + encodeURIComponent(img.src) +
                              '&bottom_left_lat=' + bottomLeft.lat +
                              '&bottom_left_lng=' + bottomLeft.lng +
                              '&top_right_lat=' + topRight.lat +
                              '&top_right_lng=' + topRight.lng
                    }})
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error('Network response was not ok');
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        // Update validity text box
                        document.getElementById('pathValidity').textContent = data.is_valid ? 'Yes' : 'No';

                        // Download the segmented mask
                        fetch(data.mask_url)
                        .then(response => response.blob())
                        .then(blob => {{
                            var url = window.URL.createObjectURL(blob);
                            var link = document.createElement('a');
                            link.href = url;
                            link.download = 'segmented_mask.png';
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            window.URL.revokeObjectURL(url);
                        }});
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        alert('Error processing image: ' + error);
                    }});
                }});
            }});

            document.getElementById('analyzeButton').addEventListener('click', function() {{
                var bounds = map.getBounds();
                var bottomLeft = bounds.getSouthWest();
                var topRight = bounds.getNorthEast();
                console.log("Current viewport for analysis:");
                console.log("Bottom-Left:", bottomLeft.lat.toFixed(6) + ", " + bottomLeft.lng.toFixed(6));
                console.log("Top-Right:", topRight.lat.toFixed(6) + ", " + topRight.lng.toFixed(6));
                console.log("Zoom Level:", map.getZoom());

                leafletImage(map, function(err, canvas) {{
                    var img = document.createElement('img');
                    var dimensions = map.getSize();
                    img.width = dimensions.x;
                    img.height = dimensions.y;
                    img.src = canvas.toDataURL();
                    var imageData = img.src;
                    fetch('/process_image/', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                        body: 'image_data=' + encodeURIComponent(imageData)
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.error) {{
                            console.error('Error processing image:', data.error);
                            alert('Error processing image: ' + data.error);
                        }} else {{
                            alert(data.message);
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error sending image:', error);
                        alert('Error sending image: ' + error);
                    }});
                }});
            }});
        </script>
        </body>
        </html>
    """

    return html


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
