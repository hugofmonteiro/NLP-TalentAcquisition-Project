import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import pandas as pd

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('../data/potential-talents - Aspiring human resources - seeking human resources.csv')

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Tokenization and Encoding for BERT
def bert_encode(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state.mean(dim=1).squeeze()

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Define key phrases for first FIT assessment
key_phrases = ["aspiring human resources", "seeking human resources"]

# Compute BERT embeddings for key phrases
key_phrase_embeddings = [bert_encode(phrase) for phrase in key_phrases]

# Calculate similarity for job titles
df['tokenized_titles'] = df['job_title'].apply(lambda x: x.lower())
for idx, phrase in enumerate(key_phrases):
    phrase_embedding = key_phrase_embeddings[idx]
    df['similarity_to_' + phrase.replace(" ", "_")] = df['tokenized_titles'].apply(
        lambda x: cosine_similarity(bert_encode(x), phrase_embedding)
    )

# Calculate similarity for 'location', so chose a city based on stared candidate
df['tokenized_locations'] = df['location'].apply(lambda x: x.lower())
chicago_embedding = bert_encode("Chicago, Illinois")
df['similarity_to_Chicago_Illinois'] = df['tokenized_locations'].apply(
    lambda x: cosine_similarity(bert_encode(x), chicago_embedding)
)

def connection_to_prob(connection):
    if connection.strip() == '500+':
        return 1.00
    else:
        # Remove any non-numeric characters before conversion
        numeric_connection = ''.join(filter(str.isdigit, connection))
        return min(int(numeric_connection) / 500, 1.00)

df['connection_prob'] = df['connection'].apply(connection_to_prob)

# Calculate probabilities based on FIT, FIT plus Connections, or even FIT plus Connections adjusted to similar city to stared candidated
df['FIT'] = df['similarity_to_aspiring_human_resources'] + df['similarity_to_seeking_human_resources']
df['FIT_AND_CONNECTIONS'] = df['similarity_to_aspiring_human_resources'] + df['similarity_to_seeking_human_resources'] + df['similarity_to_Chicago_Illinois']
df['ALL_Ajusted_Location'] = df['FIT_AND_CONNECTIONS'] + df['connection_prob']

# Candidates can be ranked based on FIT, FIT plus Connections, or even FIT plus Connections adjusted to similar city to stared candidated
df_ranked = df.sort_values(by='FIT', ascending=False)
df_ranked
