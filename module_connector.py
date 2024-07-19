import bert_embedding_extractor as be

if __name__ == "__main__":
    transcript = "Your transcript text goes here."
    words_to_contextualize = ["Your", "text", "here"]

    embeddings = be.get_contextualized_bert_embeddings(transcript, words_to_contextualize)

    # Print embeddings for each word
    for word, embedding in embeddings:
        print(f"Word: {word}, Embedding: {embedding[:5]}...")  # Print first 5 values for brevity
