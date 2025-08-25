# External library imports
from datetime import datetime
import gradio as gr

# Internal imports
from src.visual_synthesizer import VisualSynthesizer
from src.text_synthesizer import TextSynthesizer
# from src.audio_synthesizer import AudioSynthesizer
from utils.config import *    
# from utils.logger import setup_logger
from utils.helpers import richify_prompt, save_caption, save_image


def compose(prompt: str, filename: str = "generated_post"):
    """
    Main function to compose an Instagram post from a given prompt.
    
    Args:
        prompt (str): The text prompt to generate the Instagram post.
    """
    # Generate a timestamp for the filename
    # This is useful for ensuring unique filenames and tracking when the post was created
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{filename}"
    
    # Initialize the visual synthesizer
    image_gen = VisualSynthesizer()    
    # Generate the image
    image = image_gen.generate_image(prompt=richify_prompt(prompt))
    # Save the image
    image_path = save_image(image, filename=filename)
    print(f"Image saved at: {image_path}")
    # Create a caption for the post
    text_gen = TextSynthesizer()
    caption = text_gen.generate_caption(prompt=prompt)
    # Save the caption
    caption_path = save_caption(caption, filename=filename)
    print(f"Caption saved at: {caption_path}")
    return image_path, caption


if __name__ == '__main__':
    iface = gr.Interface(
        fn=compose,
        inputs=gr.Textbox(lines=5, label="Prompt", placeholder="Enter your prompt here..."),
        outputs=[
            gr.Image(type="filepath", label="Generated Image"),
            gr.Textbox(label="Generated Caption")
        ],
        title="From Words to Reels",
        description="Enter a prompt to generate an image and a corresponding social media caption.",
        allow_flagging="never"
    )
    
    # Launch the Gradio app
    iface.launch()

    # print(f"From words to reels, creates instagramable posts for your prompts")
    # # setup_logger()
    # input_prompt = input("Enter your prompt: ")
    # if not input_prompt:
    #     print("No prompt provided. Using default prompt.")
    #     input_prompt = (
    #         "Cosmos and the Universe, a vast expanse of stars and galaxies, " 
    #         "a reminder of our place in the universe. The beauty of the cosmos is "
    #         "a source of inspiration and wonder, a reminder that we are part of something much larger than ourselves." 
    #         "The universe is a canvas, painted with the colors of creation, a masterpiece that continues to unfold before our eyes."
    #     )
    # input_prompt = "Indeed, with hardship comes ease. - Quran 94:6"
    
    # # Compose a post given the prompt
    # compose(prompt=input_prompt)
    # print(f"Composition successfull. Check the output directory for the generated image and caption.")
