from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests

# Step 1: Load the processor and model
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Step 2: Load an image
url = "https://www.elle.vn/wp-content/uploads/2024/05/23/588490/FALL-24-DENIM-DIOR-OBLIQUE-CAMPAIGN-WITH-HAERIN-%C2%A9-Tanya-Zhenya-Posternak-3.jpg"  # Replace with your image URL or local path
image = Image.open(requests.get(url, stream=True).raw)

# Step 3: Process the image and generate a caption
inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(pixel_values=inputs.pixel_values)

# Step 4: Decode and print the caption
caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Caption:", caption)
