"""
Configuration variables to centralize parameters and paths.
"""
OUTPUT_DIR = "outputs/"  # Directory to save generated image and captions

# Image generation settings
IMAGE_SIZE = (512, 512)  # Size of the generated images

# Model names for easy change and reuse
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_MODEL_NAME = "microsoft/phi-2"
AUDIO_MODEL_NAME = ""  # Placeholder for audio model, can be set later

# Stable Diffusion model and device to run on
IMG_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
VIDEO_MODEL_NAME = "cerspense/zeroscope_v2_XL"  # Placeholder for video model
# Other models to try # Qwen/Qwen-Image # CompVis/stable-diffusion-v1-4
# "segmind/SSD-1B" # Or "kandinsky-community/kandinsky-3", "warp-ai/wuerstchen"
# Video generation models # cerspense/zeroscope_v2_576w # Wan‑Video/Wan2.1
DEVICE = "cuda"  # Change to "cpu" if no GPU available

# Font path for overlay text
# FONT_PATH = "./fonts/arial.ttf"
# FONT_SIZE = 40