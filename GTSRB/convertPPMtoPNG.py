import os
from PIL import Image

def convert_ppm_to_png(base_folder):
    """
    Converts all .ppm images in the 'train' and 'test' subfolders of the base folder to .png format.
    
    Args:
        base_folder (str): The path to the base folder containing 'train' and 'test' subfolders.
    """
    subfolders = ['train', 'val']
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        if not os.path.exists(folder_path):
            print(f"Subfolder '{subfolder}' does not exist in the base folder.")
            continue
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ppm'):
                # Full path of the input file
                ppm_path = os.path.join(folder_path, file_name)
                
                # Full path of the output file with .png extension
                png_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.png')
                
                try:
                    # Open the .ppm image and save it as .png
                    with Image.open(ppm_path) as img:
                        img.save(png_path, 'PNG')
                    print(f"Converted: {ppm_path} -> {png_path}")
                    
                    # Optional: Remove the original .ppm file after conversion
                    os.remove(ppm_path)
                except Exception as e:
                    print(f"Error converting {ppm_path}: {e}")

if __name__ == "__main__":
    # Change this to the path of your base folder
    base_images_folder = "dataset/images"
    convert_ppm_to_png(base_images_folder)
