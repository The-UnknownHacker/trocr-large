from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import gradio as gr 

# App title
title = "Welcome to Your First Handwritten Recognition App!"

# Load the model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

# Prediction function for handwriting
def predict(image_url, img_draw, img_upload):
    # Fetch the image from URL, handwritten canvas, or uploaded image
    if image_url:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    elif isinstance(img_draw, Image.Image):  # Ensure img_draw is a PIL Image
        image = img_draw.convert("RGB")
    elif isinstance(img_upload, Image.Image):  # Ensure img_upload is a PIL Image
        image = img_upload.convert("RGB")
    else:
        return "No valid image provided."

    # Predict the image using the microsoft/trocr-large-handwritten model loaded earlier
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Image URL", placeholder="Enter the URL of the image"),
        gr.Sketchpad(label="Draw Here", type="pil"),
        gr.Image(label="Upload Image", type="pil")
    ],
    outputs="text",
    title=title
)

interface.launch(server_name="0.0.0.0", server_port=8080)
