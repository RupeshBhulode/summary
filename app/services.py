from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Load the model once
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def get_embeddings(sentences):
    embeddings = model.encode(sentences)
    return embeddings

def get_top_questions(sentences, embeddings, num_clusters=20):
    # Clustering
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_centers = clustering_model.cluster_centers_

    # For each cluster, pick the question closest to the cluster center
    representative_questions = []

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_assignment == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]

        # Compute distances to the cluster center
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]

        representative_questions.append(sentences[closest_index])

    return representative_questions

def cluster_comments(sentences, num_clusters=20):
    embeddings = get_embeddings(sentences)
    top_questions = get_top_questions(sentences, embeddings, num_clusters)
    return top_questions
