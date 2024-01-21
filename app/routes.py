from app import model

from fastapi import APIRouter, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from io import BytesIO
import io
import tempfile
import os
import time
# import asyncio
import concurrent.futures
from datetime import datetime, timezone, timedelta
import cv2

router = APIRouter()
load_dotenv()
IST = timezone(timedelta(hours=5, minutes=30))
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get('/healthcheck')
async def healthcheck():
    return {"status": "ok"}


@router.post("/process_and_predict")
async def process_and_predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        
        current_time_in_ist = datetime.now(IST)

        filename = f"{current_time_in_ist}_captured.jpeg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(file_path, format="JPEG")

        
        final_image, total_count = draw_predictions_on_image(image, current_time_in_ist)

        img_byte_array = BytesIO()
        final_image.save(img_byte_array, format="JPEG")

        return Response(content=img_byte_array.getvalue(), headers={"detections": str(total_count), 'access-control-expose-headers': 'detections'}, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def crop_image_into_patches(img):
    width, height = img.size

    patches = []
    for i in range(0, width, 640):
        for j in range(0, height, 640):
            box = (i, j, i + 640, j + 640)
            patch = img.crop(box)
            patches.append((box, patch))

    return patches

# def draw_predictions_on_image(img, image_name):
#     patches = crop_image_into_patches(img)
#     final_image = img
#     draw = ImageDraw.Draw(final_image)
#     total_predictions = 0
#     for box, patch in patches:
#         temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
#         patch.save(temp_file.name)
#         temp_file.close()
#         prediction = model.predict(temp_file.name, confidence=25, overlap=55).json()
#         for pred in prediction.get('predictions', []):
#             x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
#             center_x, center_y = x + box[0], y + box[1]
#             draw.ellipse([(center_x - 2, center_y - 2), (center_x + 2, center_y + 2)], fill='red')
#             total_predictions += 1
#         os.unlink(temp_file.name)
    
#     return final_image, total_predictions

def predict_on_patch(patch):
    box, patch_img = patch

    # Save the patch as a temporary image file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
    patch_img.save(temp_file.name)
    temp_file.close()

    # Predict on the patch
    prediction = model.predict(temp_file.name, confidence=20, overlap=65).json()

    # Delete the temporary file
    os.unlink(temp_file.name)

    return box, prediction.get('predictions', [])

# Function to draw predictions on the final image and save as XML
def draw_predictions_on_image(image_path, current_time_in_ist):
    start_time = time.time()
    patches = crop_image_into_patches(image_path)
    final_image = image_path
    draw = ImageDraw.Draw(final_image)

    total_predictions = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Predict on patches in parallel
        futures = [executor.submit(predict_on_patch, patch) for patch in patches]

        for future in concurrent.futures.as_completed(futures):
            box, predictions = future.result()

            # Draw predictions on the final image and create XML annotations
            for pred in predictions:
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                center_x, center_y = x + box[0], y + box[1]  # Calculate center coordinates

                # Draw a dot at the center
                draw.ellipse([(center_x - 2, center_y - 2), (center_x + 2, center_y + 2)], fill='red')
                total_predictions += 1  # Increment total predictions
    
    end_time = time.time()

    # Print total time taken
    total_time_taken = end_time - start_time
    print(f"Total time taken: {total_time_taken:.2f} seconds")


    # Add text to the top left corner with total count
    # text = f"Total Count: {total_predictions}"
    # font_size = 250
    # font_color = (255, 0, 0)
    # draw.text((10, 10), text, font=None, fill=font_color)
    filename = f"{current_time_in_ist}_predicted_{total_predictions}.jpeg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    final_image.save(file_path, format='JPEG')


    return final_image, total_predictions

