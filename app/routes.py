from app import model

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import io
import tempfile
import os
import xml.etree.ElementTree as ET

router = APIRouter()
load_dotenv()

@router.get('/healthcheck')
async def healthcheck():
    return {"status": "ok"}


@router.post("/process_and_predict")
async def process_and_predict(file: UploadFile = File(...)):
    try:
        # Open the uploaded image file
        image = Image.open(BytesIO(await file.read()))
        
        # Process the image using your custom function
        # processed_image = process_image(image)
        final_image, total_count, xml_content = draw_predictions_on_image(image, "robo_flow_image.png")

        # Convert the processed image to bytes
        img_byte_array = BytesIO()
        final_image.save(img_byte_array, format="PNG")
        # encoded_image_string = base64.b64encode(img_byte_array.getvalue()).decode("utf-8")
        # image_bytes: bytes = final_image.tobytes()

        additional_data = {"xml_data": xml_content, "detections": str(total_count), "status": "success", "message": "Image processed successfully"}

        # # Combine image and JSON data in a dictionary
        # return {"mime" : "image/png","image": encoded_image_string, **additional_data}
        # return Response(content = final_image, media_type="image/png")
        # return Response(content=img_byte_array.getvalue(), headers = additional_data, media_type="image/png")

        # # Return the response with both image and JSON data
        return StreamingResponse(content=io.BytesIO(img_byte_array.getvalue()), media_type="image/png")
        # return additional_data

        # image_response = StreamingResponse(content=io.BytesIO(img_byte_array.getvalue()), media_type="image/png")

        # json_response = JSONResponse(content=additional_data)

        # # Combine the responses into a tuple
        # responses = (image_response, json_response)

        # # Return the tuple of responses
        # return responses

    except Exception as e:
        # Handle exceptions, e.g., if the uploaded file is not an image
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def crop_image_into_patches(img):
    # img = Image.open(image_path)
    width, height = img.size

    patches = []
    for i in range(0, width, 640):
        for j in range(0, height, 640):
            box = (i, j, i + 640, j + 640)
            patch = img.crop(box)
            patches.append((box, patch))  # Store both box and patch

    return patches

def draw_predictions_on_image(img, image_name):
    # Get patches from the input image
    patches = crop_image_into_patches(img)
    
    # Create a new image to draw predictions on
    # final_image = Image.open(image_path)
    final_image = img
    draw = ImageDraw.Draw(final_image)
    
    total_predictions = 0
    
    annotations = ET.Element("annotations")
    
    for box, patch in patches:
        # Save the patch as a temporary image file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        patch.save(temp_file.name)
        temp_file.close()
        
        # Predict on each patch
        prediction = model.predict(temp_file.name, confidence=25, overlap=10).json()
        
        # Draw predictions on the final image and create XML annotations
        for pred in prediction.get('predictions', []):
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            xmin, ymin = max(0, x + box[0] - w), max(0, y + box[1] - h)  # Ensure non-negative values
            xmax, ymax = x + box[0] + w, y + box[1] + h  # Normal xmax and ymax
            
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
            
            # Create XML annotations
            annotation = ET.SubElement(annotations, "annotation")
            ET.SubElement(annotation, "filename").text = image_name
            ET.SubElement(annotation, "object").text = "prediction"
            ET.SubElement(annotation, "xmin").text = str(xmin)
            ET.SubElement(annotation, "ymin").text = str(ymin)
            ET.SubElement(annotation, "xmax").text = str(xmax)
            ET.SubElement(annotation, "ymax").text = str(ymax)
           
            total_predictions += 1  # Increment total predictions
        
        # Delete the temporary file
        os.unlink(temp_file.name)
    
    # Save XML file
    xml_tree = ET.ElementTree(annotations)
    xml_content = BytesIO()
    xml_tree.write(xml_content, encoding="utf-8", xml_declaration=True)
    # return xml_content.getvalue().decode("utf-8")
    # xml_tree.write("IMG20231227152657.xml")
    
    return final_image, total_predictions, xml_content.getvalue().decode("utf-8")


