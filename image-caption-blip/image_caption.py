import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# It can generate text based on an input image and an optional piece of text. 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
img_path = "happy_dog.jpg"

# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the caption
print(caption)