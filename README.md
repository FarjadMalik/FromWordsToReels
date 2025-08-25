# From Words to Reels

This project generates social media posts, including an image and a caption, from a user-provided text prompt. It leverages deep learning models for both text-to-image synthesis and text generation to create engaging content.

## How it Works

The process is orchestrated by the `main.py` script and follows these steps:

1.  **User Input**: The script prompts the user to enter a text prompt.
2.  **Image Generation**: The `VisualSynthesizer` takes the prompt, enhances it, and uses a text-to-image diffusion model (e.g., Stable Diffusion) to generate a corresponding image.
3.  **Caption Generation**: The `TextSynthesizer` uses the original prompt to generate a suitable caption for the post using a causal language model.
4.  **Output**: Both the generated image (`.png`) and the caption (`.txt`) are saved to the `outputs/` directory, prefixed with a timestamp.

## Project Structure

```
.
├── main.py                 # Main script to run the application
├── README.md               # This file
├── outputs/                # Directory for generated images and captions
├── src/
│   ├── visual_synthesizer.py # Handles image generation
│   ├── text_synthesizer.py   # Handles text/caption generation
└── utils/
    ├── config.py             # Configuration for models and paths
    └── helpers.py            # Helper functions for saving files etc.
```

## Setup and Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

2.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    torch
    diffusers
    transformers
    sentence-transformers
    Pillow
    accelerate
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To generate a post, run the `main.py` script:

```bash
python main.py
```

You will be prompted to enter your text. After processing, the generated image and caption will be saved in the `outputs` directory.

## Configuration

You can customize the models and other parameters by editing the `utils/config.py` file. This allows you to easily swap out different text-to-image or language models.
