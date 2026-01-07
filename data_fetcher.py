import os
import argparse
import time
import math
import requests
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from src.config import cfg

def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to title coordinates for the given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x,y

def esri_url(lat, lon, zoom):
    """
    Generate ESRI World Imagery tile URL (FREE - no API key needed).
    Uses ArcGIS World Imagery service.
    """

    x, y = lat_lon_to_tile(lat, lon, zoom)
    return (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{}/{}/{}".format(zoom, y, x)
    )

def fetch_image(lat, lon, max_retries=5, retry_delay=2):
    """
    Fetch satellite image from ESRI World Imagery (FREE).
    Downloads a title and returns it as PNG bytes.
    Includes retry logic for network errors.
    Uses OpenCV for image processing.
    """
    url = esri_url(lat, lon, cfg.zoom)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36"
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30, headers=headers)
            if r.status_code == 200:
                img_array = np.frombuffer(r.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    print("Failed to decode image")
                    return None

                # Resize to desired title size using OpenCV 
                img = cv2.resize(img, (256, 256))

                # Encode back to PNG bytes using OpenCV
                success, buffer = cv2.imencode('.png', img)
                if success: 
                    return buffer.tobytes()
                return None
            elif r.status_code == 429: #Rate limited
                wait_time = retry_delay * (attempt + 1)
                time.sleep(wait_time)
                continue
            else:
                print(f"Failed to fetch image: HTTP {r.status_code}")
                return None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    print(f"Error fetching image after {max_retries} attempts")
    return None


def download(df, lat_col="lat", lon_col="long", id_col="id"):
    """Download satellite images for dataframe rows"""
    os.makedirs(cfg.image_dir, exist_ok=True)
    paths = {}
    skipped = 0
    downloaded = 0
    failed_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading satellite tiles"):
        lat = row[lat_col]
        lon = row[lon_col]
        pid = row[id_col]
        fname = f"{pid}_{lat:.5f}_{lon:.5f}.png"
        fpath = os.path.join(cfg.image_dir, fname)

        # Check cache - skip if Already downloaded
        if os.path.exists(fpath):
            paths[pid] = fpath
            skipped += 1
            continue

        content = fetch_image(lat, lon)
        if content is None:
            failed_list.append((lat, lon, pid, fpath))
            continue

        with open(fpath, "wb") as f:
            f.write(content)

        paths[pid] = fpath
        downloaded += 1
        time.sleep(0.1)  # polite pause

    # Retry failed downloads
    retry_success = 0
    if failed_list:
        print(f"\nRetrying {len(failed_list)} failed downloads...")
        time.sleep(2) # Wait before retrying
        retry_success = 0

        for lat, lon, pid, fpath in tqdm(failed_list, desc="Retrying failed"):
            time.sleep(0.5) #slower pace for retries
            content = fetch_image(lat, lon, max_retries=3, retry_delay=3)
            if content:
                with open(fpath, "wb") as f:
                    f.write(content)
                paths[pid] = fpath
                retry_success += 1

        final_failed = len(failed_list) - retry_success
        print(f"Retry results: {retry_success} recovered, {final_failed} still failed")
    else:
        final_failed = 0

#summary
    print(f"\n{ '='*50}")
    print(f"Download Summary:")
    print(f" - Already cached (skipped): {skipped}")
    print(f" - Newly downloaded: {downloaded}")
    print(f"- Recovered on retry: {retry_success}")
    print(f" - Failed: {final_failed}")
    print(f" - Total images: {skipped + downloaded}")
    print(f" - Saved to: {cfg.image_dir}")
    print(f"{'=' * 50}")

    return paths

def main():
    train_df = pd.read_excel(cfg.train_xlsx)
    test_df = pd.read_excel(cfg.test_xlsx)
    download(pd.concat([train_df, test_df], axis=0))


# RUN DIRECTLY
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()

# Example CSV load (REMOVE BEFORE SUBMISSION)
# df = pd.read_csv("data/coordinates.csv")  
# download(df)

