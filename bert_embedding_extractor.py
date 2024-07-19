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

def get_contextualized_bert_embeddings(text: str, words: List[str], max_length: int = 512) -> List[Tuple[str, torch.Tensor]]:
    """
    Get contextualized BERT embeddings for specified words in the given text.

    Parameters:
    text (str): The input text.
    words (List[str]): The list of words to get embeddings for.
    max_length (int): The maximum length of the input sequence for BERT (default is 512).

    Returns:
    List[Tuple[str, torch.Tensor]]: A list of tuples containing the word and its embedding.
    """
    # Tokenize the text using spaCy
    doc = nlp(text)
    
    # Split the text into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    for token in doc:
        token_length = len(tokenizer.encode(token.text, add_special_tokens=False))
        if current_length + token_length >= max_length - 2:  # Subtract 2 for special tokens
            chunks.append(current_chunk)
            current_chunk = [token]
            current_length = token_length
        else:
            current_chunk.append(token)
            current_length += token_length
    if current_chunk:
        chunks.append(current_chunk)
    
    # Process each chunk
    token_embeddings = []
    for chunk in chunks:
        sentence = " ".join([token.text for token in chunk])
        
        # Tokenize the chunk using BERT tokenizer
        bert_tokens = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
        
        # Check if the bert_tokens length exceeds max_length
        if bert_tokens.input_ids.size(1) > max_length:
            continue
        
        # Get BERT embeddings for the chunk
        with torch.no_grad():
            outputs = model(**bert_tokens)
        
        # The embeddings are in the outputs.last_hidden_state tensor
        embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
        
        # Map spaCy tokens to BERT tokens and extract their embeddings
        for token in chunk:
            if token.text not in words:
                continue
            
            # Get the index of the token in the BERT token list
            bert_token_indices = tokenizer.encode(token.text, add_special_tokens=False)

            # Find the BERT token index in the chunk
            try:
                start = bert_tokens.input_ids.squeeze(0).tolist().index(bert_token_indices[0])
                end = start + len(bert_token_indices)
            except ValueError:
                continue

            # Average the embeddings of the subword tokens
            word_embedding = embeddings[start:end].mean(dim=0)
            token_embeddings.append((token.text, word_embedding))
    
    return token_embeddings
