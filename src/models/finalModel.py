import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import pandas as pd

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

def cosine_similarity(vec1, vec2):
    vec1 = vec1.view(-1)
    vec2 = vec2.view(-1)
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Defining key phrases for first FIT assessment
key_phrases = ["aspiring human resources", "seeking human resources"]

# Computing BERT embeddings for key phrases
key_phrase_embeddings = [bert_encode(phrase) for phrase in key_phrases]

# Calculating similarity for job titles
df['tokenized_titles'] = df['job_title'].apply(lambda x: x.lower())
for idx, phrase in enumerate(key_phrases):
    phrase_embedding = key_phrase_embeddings[idx]
    df['similarity_to_' + phrase.replace(" ", "_")] = df['tokenized_titles'].apply(
        lambda x: cosine_similarity(bert_encode(x), phrase_embedding).item()
    )

# Calculating average of similarities
df['AVG_cosine_similarity'] = (df['similarity_to_aspiring_human_resources'] + df['similarity_to_seeking_human_resources'])/2

# Ranking candidates based on most similar to keywords
df_ranked = df.sort_values(by='AVG_cosine_similarity', ascending=False)
df_ranked

# Reranking based on job_title of stared candidate
starred_candidate_job_title = "Human Resources Coordinator"

# Calculating BERT embedding
starred_job_title_embedding = bert_encode(starred_candidate_job_title)

# Modifying key phrase embeddings based on the starred candidate
new_key_phrase_embeddings = [(bert_encode(phrase) + starred_job_title_embedding) / 2 for phrase in key_phrases]  # Average method

# Recalculating the similarities with new embeddings
for idx, phrase in enumerate(key_phrases):
    new_phrase_embedding = new_key_phrase_embeddings[idx]
    df['new_similarity_to_' + phrase.replace(" ", "_")] = df['tokenized_titles'].apply(
        lambda x: cosine_similarity(bert_encode(x), new_phrase_embedding).item()
    )

# Update the average fit score
df['AVG_cosine_similarity_reranked'] = df[['new_similarity_to_' + phrase.replace(" ", "_") for phrase in key_phrases]].mean(axis=1)

# Rerank the candidates based on the new fit score
df_reranked = df.sort_values(by='AVG_cosine_similarity_reranked', ascending=False)
df_reranked