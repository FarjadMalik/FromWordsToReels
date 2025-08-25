import os 
from datetime import datetime
from utils.config import OUTPUT_DIR


def save_caption(caption: str, filename: str, output_dir: str = OUTPUT_DIR):
    """
    Save the generated caption to a text file.

    Args:
        caption (str): The generated  text.
        filename (str): Optional. The filename to use (without extension).
        output_dir (str): Folder where the file will be saved. Defaults to 'outputs'.

    Returns:
        str: Full path to the saved file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if not filename:
        raise ValueError("Filename must be provided")
    if not filename.endswith('.txt'):
        filename += '.txt'

    filepath = os.path.join(output_dir, filename)

    # Save the caption
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(caption.strip())

    return filepath


def save_image(image, filename: str, output_dir: str = OUTPUT_DIR):
    """
    Saves the generated image to the specified directory.

    Args:
        image: The generated image to save.
        output_dir (str): The directory where the image will be saved.
        filename (str): The name of the file to save the image as.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    # Generate filename if not provided
    if not filename:
        raise ValueError("Filename must be provided")
    if not filename.endswith('.png'):
        filename += '.png'
    # Construct the full path
    image_path = os.path.join(output_dir, filename)
    image.save(image_path)
    return image_path


def richify_prompt(text: str) -> str:
    """
    Beautifies the input text by removing extra spaces and ensuring proper formatting.
    
    Args:
        text (str): The input text to be beautified.
        
    Returns:
        str: The beautified text.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    text_prompt = (
        f"(best quality:1.3), (intricate details:1.2), high-resolution digital painting of {text}, "
        "ArtStation fine art"
    )
    return ' '.join(text_prompt.split()).strip() if text_prompt else ''


# More richify prompts
# image_prompt = f"A beautiful and artistic representation of the following text: '{text}'; in the style of Studio Ghibli, digital art, 4k, vibrant colors, intricate details, Artstation."
# Epic Cinematic Illustration
# image_prompt = f"(best quality:1.4), (masterpiece:1.3), (detailed:1.2), 4k, wide-angle cosmic panorama of the Big Bang and expanding universe transitioning into the creation of life on Earth, poetic illumination, vibrant nebulae and galaxies, in the style of Studio Ghibli and ArtStation concept art, divine origins, dramatic lighting, awe‑inspiring mood"
# Realistic Documentary Style
# image_prompt = f"(realistic cosmic time-lapse:1.2), (masterpiece:1.2), ultra-detailed 8k scientific illustration of cosmic evolution from the Big Bang to modern civilization, expanding galaxies, formation of Earth, emergence of life, soft ambient lighting, realistic textures, wide-angle shot, inspired by ArtStation and nature documentaries, contemplative mood"
# Animated Spiritual Universe
# image_prompt = f"(best quality:1.3), (intricate details:1.2), high-resolution digital painting of the universe expanding from the Big Bang into Earth’s formation, evolving life and early civilization, soft celestial lighting, pastel and vibrant colors, in the style of Studio Ghibli animation, ArtStation fine art, uplifting and mystical atmosphere, panoramic composition"
