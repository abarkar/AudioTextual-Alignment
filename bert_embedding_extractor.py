"""
BERT Embedding Extraction Module
================================

This module provides a function to extract contextualized BERT embeddings
for specified words in a given text using spaCy for tokenization and the
Hugging Face Transformers library for BERT embeddings.

Functions
---------
get_contextualized_bert_embeddings(text: str, words: List[str]) -> List[Tuple[str, torch.Tensor]]
    Get contextualized BERT embeddings for specified words in the given text.

Example Usage
-------------
if __name__ == "__main__":
    transcript = "Your transcript text goes here."
    words_to_contextualize = ["Your", "text", "here"]

    embeddings = get_contextualized_bert_embeddings(transcript, words_to_contextualize)

    # Print embeddings for each word
    for word, embedding in embeddings:
        print(f"Word: {word}, Embedding: {embedding[:5]}...")  # Print first 5 values for brevity
"""
import spacy
from transformers import BertTokenizer, BertModel
import torch
from typing import List, Tuple

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_contextualized_bert_embeddings(text: str, words: List[str]) -> List[Tuple[str, torch.Tensor]]:
    """
    Get contextualized BERT embeddings for specified words in the given text.

    Parameters:
    text (str): The input text.
    words (List[str]): The list of words to get embeddings for.

    Returns:
    List[Tuple[str, torch.Tensor]]: A list of tuples containing the word and its embedding.
    """
    # Tokenize the text using spaCy
    doc = nlp(text)
    
    # Join the words to form the sentence
    sentence = " ".join([token.text for token in doc])
    
    # Tokenize the entire sentence using BERT tokenizer
    bert_tokens = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
    
    # Get BERT embeddings for the entire sentence
    with torch.no_grad():
        outputs = model(**bert_tokens)
    
    # The embeddings are in the outputs.last_hidden_state tensor
    embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    
    # Map spaCy tokens to BERT tokens and extract their embeddings
    token_embeddings = []
    for word in words:
        # Find the token in the spaCy doc
        token = next((token for token in doc if token.text == word), None)
        if token is None:
            continue

        # Get the index of the token in the BERT token list
        bert_token_indices = tokenizer.encode(token.text, add_special_tokens=False)

        # Find the BERT token index in the sentence
        try:
            start = bert_tokens.input_ids.squeeze(0).tolist().index(bert_token_indices[0])
            end = start + len(bert_token_indices)
        except ValueError:
            continue

        # Average the embeddings of the subword tokens
        word_embedding = embeddings[start:end].mean(dim=0)
        token_embeddings.append((token.text, word_embedding))
    
    return token_embeddings
