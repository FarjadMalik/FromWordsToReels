import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Importing the necessary configuration for model names
from utils.config import TEXT_MODEL_NAME, EMBEDDING_MODEL_NAME


class TextSynthesizer:
    def __init__(self, embed_model=EMBEDDING_MODEL_NAME, text_model=TEXT_MODEL_NAME): # TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/phi-2 HuggingFaceH4/zephyr-1.1B-alpha model_name: str = "gpt-3.5-turbo"
        """
        Initializes the TextAnalyzer with a specified sentence-transformer model.

        Args:
            model_name (str): The name of the sentence-transformer model to use.
        """
        self.model = SentenceTransformer(embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model,
            device_map="auto",
            torch_dtype="auto"
        )
        
        self.text_generator = pipeline(
            'text-generation',
            model=text_model,
            tokenizer=self.tokenizer
        )

    def get_embedding(self, text: str):
        """
        Generates an embedding for the input text.

        Args:
            text (str): The input text (e.g., a quote, poem, or verse).

        Returns:
            numpy.ndarray: A vector embedding of the text.
        """
        return self.model.encode(text)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text by removing unwanted characters and trimming extra whitespace.
        """
        # Remove any characters except letters, numbers, punctuation, and basic symbols
        cleaned = re.sub(r"[^\w\s.,:;!?'\"-]", "", text)
        # Normalize whitespace to single spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


    def validate_text_length(self, text: str, max_length: int = 300) -> bool:
        """
        Validate text length to avoid overly long inputs.
        Returns True if valid, False otherwise.
        """
        return 0 < len(text) <= max_length

    def extract_keywords(self, text: str) -> list:
        """
        Extracts keywords from the given text.
        This is a placeholder and can be enhanced with more sophisticated NLP techniques
        or another LLM for semantic keyword extraction.

        Args:
            text (str): The input text.

        Returns:
            list: A list of extracted keywords.
        """
        # Simple example: convert to lowercase and split by whitespace.
        # For production, consider using NLTK, SpaCy, or an LLM-based keyword extractor.
        return [word for word in text.lower().split() if len(word) > 2] # Basic filtering
    
    def generate_caption(self, prompt: str, max_new_tokens: int = 300) -> str:
        """
        Generates a caption based on the provided prompt, enriched with poetic or authoritative quotes.
        
        :param prompt: The input text prompt for text generation.
        :param max_length: The maximum length of the generated caption.
        :return: The generated caption.
        """
        # Craft a more detailed prompt for the model to generate a fitting caption
        generation_prompt = (
            f"Write a detailed, poetic, and informative paragraph about the following topic: \n'{prompt}'.\n"
            f"Use vivid, emotional language and include relevant verses or quotes by poets, philosophers, or scientists."
            f"The paragraph should be knowledgable, well researched and engaging. The tone should be educational and inspirational, not casual or conversational."
            f"Dont use emojis or hastags or words like response or answer, just write the paragraph directly.\n"
        )

        generated_outputs = self.text_generator(
            generation_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,   # <-- helps reduce loops
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        raw_output = generated_outputs[0]['generated_text']
        
        # Remove the initial prompt from the result
        if generation_prompt in raw_output:
            # The model will return the prompt plus the generated text, so we clean it up.
            # We find the generated part by removing the initial prompt.
            caption = raw_output.split(generation_prompt)[-1].strip()
        else:
            caption = raw_output.strip()

        # Clean up the caption to ensure it's a single coherent block
        caption = caption.replace(generation_prompt, "").strip()
        
        # Further clean the text to ensure it's a single, coherent block
        # caption = self.clean_text(caption.split('\n')[0])
        
        return caption