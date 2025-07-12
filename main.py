from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a data model
class Place(BaseModel):
    name: str
    latitude: float
    longitude: float

places = []  # Store places in memory

@app.post("/add_place")
def add_place(place: dict):
    places.append(place)
    return {"message": f"Place '{place['name']}' added at ({place['latitude']}, {place['longitude']})"}

@app.get("/get_places")
def get_places():
    return {"places": places}

@app.put("/update_places")
def update_places(placeName: str, place: dict):
    for place1 in places:
        if (place1['name']==placeName):
            place1 = place

