import spacy
from transformers import BertTokenizer, BertModel
import torch

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get contextualized BERT embeddings for each word in the text
def get_contextualized_bert_embeddings(text):
    # Tokenize the text using spaCy
    doc = nlp(text)
    print("This is your doc:", doc)
    
    # Extract words from the spaCy doc
    words = [token.text for token in doc]
    print("This is your words list:", words)

    
    # Join the words to form the sentence
    sentence = " ".join(words)
    
    # Tokenize the entire sentence using BERT tokenizer
    bert_tokens = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
    print("This is your bert tokens:", bert_tokens)

    
    # Get BERT embeddings for the entire sentence
    with torch.no_grad():
        outputs = model(**bert_tokens)
    
    # The embeddings are in the outputs.last_hidden_state tensor
    embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    print("This is your embeddings:", embeddings)
    
    # Map spaCy tokens to BERT tokens and extract their embeddings
    token_embeddings = []
    for i, token in enumerate(doc):
        print("This is your i:", i, " and token: ", token)

        # Get the index of the token in the BERT token list
        bert_token_indices = tokenizer.encode(token.text, add_special_tokens=False)
        print("This is your bert_token_indices:", bert_token_indices)

        # Find the BERT token index in the sentence
        start = bert_tokens.input_ids.squeeze(0).tolist().index(bert_token_indices[0])
        end = start + len(bert_token_indices)
        print("This is your start:", start, " and end:", end)

        # Average the embeddings of the subword tokens
        word_embedding = embeddings[start:end].mean(dim=0).numpy()
        print("This is your word_embedding:", word_embedding)

        token_embeddings.append((token.text, word_embedding))
    
    return token_embeddings

# Example usage
transcript = "Your transcript text goes here."
embeddings = get_contextualized_bert_embeddings(transcript)

# Print embeddings for each word
for word, embedding in embeddings:
    print(f"Word: {word}, Embedding: {embedding[:5]}...")  # Print first 5 values for brevity
