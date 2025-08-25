import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from typing import Optional
# Importing the model name from a configuration file
# This allows for easy changes to the model without modifying the code
# Ensure that the model_name is defined in utils/config.py
from utils.config import IMG_MODEL_NAME, VIDEO_MODEL_NAME, OUTPUT_DIR


class VisualSynthesizer:
    def __init__(self, 
                 img_model: str = IMG_MODEL_NAME,
                 video_model: str = VIDEO_MODEL_NAME):
        """
        Initializes the ImageGenerator with a specified text-to-image model.

        Args:
            img_model (str): The Hugging Face model ID for the diffusion model.
            video_model (str): The Hugging Face model ID for the video generation model (if applicable).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        torch.backends.cudnn.benchmark = True  # Optimize for input sizes

        # Initialize text-to-image pipeline with the specified model
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            img_model,
            torch_dtype=self.torch_dtype,
            variant="fp16" if self.torch_dtype == torch.float16 else None,
            low_cpu_mem_usage=True
        ).to(self.device)

        # Initialize text-to-video pipeline
        # self.video_pipe = DiffusionPipeline.from_pretrained(
        #     video_model,
        #     torch_dtype=self.torch_dtype,
        #     variant="fp16" if self.torch_dtype == torch.float16 else None,
        #     low_cpu_mem_usage=True
        # ).to(self.device)
        # self.video_pipe.enable_model_cpu_offload()
            

    def generate_image(self, prompt: str, 
                       negative_prompt: str = "blurry, distorted, poorly drawn, watermark", 
                       num_inference_steps: int = 50, guidance_scale: float = 7.5):
        image = self.image_pipe(prompt, 
                          negative_prompt=negative_prompt,
                          num_inference_steps=num_inference_steps, 
                          guidance_scale=guidance_scale
                          ).images[0]
        return image    
    
    # TODO: Fix the video generation method use the correct pipeline and parameters
    # This is a placeholder implementation, adjust as needed for your video generation requirements
    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 24,  # ~1 second at 24 fps
        fps: int = 8,
        output_path: Optional[str] = "output.mp4",
        guidance_scale: float = 12.5,
        num_inference_steps: int = 25
    ) -> str: # type: ignore
        """
        Generates a short video from a text prompt.

        Args:
            prompt (str): Text prompt to guide generation.
            negative_prompt (str): Optional negative prompts.
            num_frames (int): Number of video frames.
            fps (int): Frame rate for the video.
            output_path (str): Path to save output video.
            guidance_scale (float): Guidance scale for generation.
            num_inference_steps (int): Number of inference steps.

        Returns:
            str: Path to saved video file.
        """
        # video_output = self.video_pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     num_frames=num_frames,
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=num_inference_steps
        # ).frames

        # result = self.video_pipe(prompt, num_frames=num_frames, **kwargs)
        # frames = result.frames[0]
        # video_path = export_to_video(frames, output_video_path=f"{OUTPUT_DIR}_video", fps=fps)
        # return video_path
        pass