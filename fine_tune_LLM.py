import os
import numpy as np
from PIL import Image
import pytesseract
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig


# Define the path to the image file
image_path = '/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/jsw_tables/2490745_table.jpg'

# Open the image
receipt_image = Image.open(image_path)

# Convert the image to an array
receipt_image_array = np.array(receipt_image.convert('RGB'))

def tesseract_scan(image):
    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    boxes = []
    txts = []
    scores = []
    
    # Extract bounding boxes, text, and confidence scores
    n_boxes = len(result['level'])
    for i in range(n_boxes):
        if result['text'][i].strip():
            (x, y, w, h) = (result['left'][i], result['top'][i], result['width'][i], result['height'][i])
            boxes.append([(x, y), (x + w, y + h)])
            txts.append(result['text'][i])
            scores.append(float(result['conf'][i]))
    
    return txts, list(zip(boxes, txts, scores))

# Perform OCR scan
receipt_texts, receipt_boxes = tesseract_scan(receipt_image)
print(50 * "--", "\ntext only:\n", receipt_texts)
print(50 * "--", "\nocr boxes:\n", receipt_boxes)



# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
bnb_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# control model memory allocation between devices for low GPU resource (0,cpu)
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": 0,
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model.embed_tokens": 0,
    "model.layers":0,
    "model.norm":0    
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# model use for inference
model_id="mychen76/mistral7b_ocr_to_json_v1"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,  
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=device_map)
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


prompt=f"""### Instruction:
You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object. 
Don't make up value not in the Input. Output must be a well-formed JSON object.```json

### Input:
{receipt_boxes}

### Output:
"""

with torch.inference_mode():
    inputs = tokenizer(prompt,return_tensors="pt",truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=512) ##use_cache=True, do_sample=True,temperature=0.1, top_p=0.95
    result_text = tokenizer.batch_decode(outputs)[0]
    print(result_text)