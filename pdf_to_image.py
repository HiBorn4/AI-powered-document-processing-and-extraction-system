import os
from pdf2image import convert_from_path
from PIL import Image, ImageOps

def convert_pdfs_to_images(input_dir, output_dir, dpi=300):
    """
    Convert all PDF files in the input directory to images with the specified DPI
    and save them in the output directory. Ensures images are in the correct orientation
    and are horizontal.

    Args:
    input_dir (str): The path to the directory containing PDF files.
    output_dir (str): The path to the directory where images will be saved.
    dpi (int): The DPI (dots per inch) for the output images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file_name)
            images = convert_from_path(pdf_path, dpi=dpi)
            
            base_name = os.path.splitext(file_name)[0]
            for i, image in enumerate(images):
                # Ensure the image is in the correct orientation
                image = ImageOps.exif_transpose(image)
                
                # Rotate image to make it horizontal if it is vertical
                if image.height > image.width:
                    image = image.rotate(90, expand=True)
                
                image_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.png")
                image.save(image_path, 'PNG')
                print(f"Saved: {image_path}")

# Example usage
input_directory = "data/tata/test"
output_directory = "data/tata/new_imgs"
desired_dpi = 300

convert_pdfs_to_images(input_directory, output_directory, desired_dpi)
