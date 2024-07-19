import os
import json
import bert_embedding_extractor as be

def read_transcripts(folder_path: str):
    """
    Read all text files from the specified folder and return their content as a list of strings.

    Parameters:
    folder_path (str): The path to the folder containing transcript files.

    Returns:
    List[str]: A list of transcript texts.
    """
    transcripts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                transcripts.append(file.read())
    return transcripts

def read_words(json_file: str):
    """
    Read the JSON file and return a list of words.

    Parameters:
    json_file (str): The path to the JSON file containing words.

    Returns:
    List[str]: A list of words to contextualize.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    words = list(data.values())
    return words

if __name__ == "__main__":
    # Define paths
    transcripts_folder = './transcripts'
    words_json = './words.json'

    # Read transcripts and words
    transcripts = read_transcripts(transcripts_folder)
    words_to_contextualize = read_words(words_json)

    for transcript in transcripts:
        embeddings = be.get_contextualized_bert_embeddings(transcript, words_to_contextualize)

        # Print embeddings for each word
        for word, embedding in embeddings:
            print(f"Word: {word}, Embedding: {embedding[:5]}...")  # Print first 5 values for brevity
