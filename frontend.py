import aiohttp
import asyncio
import webbrowser
import os
import time

async def fetch_coordinates(session, url):
    try:
        async with session.get(url) as response:
            return await response.json()
    except Exception as e:
        print(f"Error fetching coordinates: {e}")
        return None

async def fetch_map_html(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching map HTML: {e}")
        return None

async def collect_route_data():
    print("Collecting route data...")
    
    async with aiohttp.ClientSession() as session:
        while True:
            data = await fetch_coordinates(session, "http://localhost:8000/get_coordinates")
            
            if not data:
                print("No data received, retrying...")
                await asyncio.sleep(1)
                continue
                
            if "status" in data and data["status"] == "end":
                print("Route data collection completed!")
                break
                
            # Print coordinates with path_index
            print(f"Received coordinates: {data['latitude']}, {data['longitude']}, Surface: {data['surface']}, Path Index: {data.get('path_index', 'Not specified')}")
            await asyncio.sleep(0.5)
        
        # Fetch and save the map
        print("Fetching final map with satellite imagery and road detection...")
        html_content = await fetch_map_html(session, "http://localhost:8000/")
        
        if html_content:
            with open("rider_path.html", "w") as f:
                f.write(html_content)
            file_path = os.path.abspath("rider_path.html")
            print(f"Map saved to: {file_path}")
            print("Opening interactive map. Use the 'Analyze Current View' button to detect unmapped roads.")
            webbrowser.open('http://localhost:8000/map')  
        else:
            print("Failed to fetch map HTML")

if __name__ == "__main__":
    asyncio.run(collect_route_data())