import logging
from typing import List
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Handles text preprocessing, tokenization, and data cleaning.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_texts(self, texts: List[str]):
        """
        Tokenizes a list of texts using the provided tokenizer.
        """
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def clean_text(self, text: str) -> str:
        """
        Cleans and preprocesses text (e.g., removes special characters, lowercasing).
        """
        text = text.lower().strip()
        return text  # Add more cleaning steps if needed

logger.info("Preprocessor initialized.")
