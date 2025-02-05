import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# It can generate text based on an input image and an optional piece of text. 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    # Generate a caption for the image
    # Decode the generated tokens to text and store it into `caption`
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

#img_path = "happy_dog.jpg"

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()
