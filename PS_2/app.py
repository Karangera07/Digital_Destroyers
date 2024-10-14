from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spacy

app = Flask(__name__)

# Load spaCy's English language model and transformer model only once
nlp = spacy.load("en_core_web_sm")
retriever_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(retriever_name)
model = AutoModel.from_pretrained(retriever_name)

# Load precomputed chunk embeddings and metadata
chunk_embeddings = np.load(fr'C:\Users\THARUN CHANDA\Desktop\AIML\PS_2\chunk_embeddings.npy')
with open(fr'C:\Users\THARUN CHANDA\Desktop\AIML\PS_2\chunks.json', 'r') as f:
    chunks = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def function_query():
    data = request.json
    query = data.get('query')

    if query is None:
        return jsonify({"error": "No query provided"}), 400

    # Function to embed text
    def embed_text(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    # Retrieve top-k relevant articles
    def retrieve_top_k_articles(query, k=3):
        query_embedding = embed_text(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_chunks = [chunks[i] for i in top_k_indices]

        articles = []
        for chunk in top_k_chunks:
            articles.append({
                'title': chunk['title'],
                'author': chunk.get('author', 'Unknown'),
                'url': chunk.get('url', '#'),
                'source': chunk['source'],
                'category': chunk['category'],
                'published_at': chunk['published_at'],
                'fact': chunk['chunk']
            })
        return articles

    # Detect query type
    def detect_query_type(query):
        if (len(query)<=10) :
          return "null_query"
        elif any(keyword in query.lower() for keyword in ["where", "which", "when", "what", "who"]):
            return "Inference_query"
        elif any(keyword in query.lower() for keyword in ["do", "does", "is", "are", "was", "were"]):
            return "Comparison_query"
        else:
            return "Temporal_query"

    # Extract named entities
    def extract_named_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "PRODUCT", "GPE", "LOC"}]
        return entities

    # Generate an answer based on the query type
    def generate_answer(query, articles, query_type):
        if query_type in ["Comparison_query", "Temporal_query"]:
            facts = [article['fact'] for article in articles]
            answer = "Yes" if len(set(facts)) == 1 else "No"
        elif query_type == "Inference_query":
            combined_chunks = " ".join([article['fact'] for article in articles])
            entities = extract_named_entities(combined_chunks)
            most_common_entity = Counter(entities).most_common(1)[0][0] if entities else "No relevant entity found"
            answer = most_common_entity
        else:
            answer = "Unknown query type"

        return answer

    # Process the query
    query_type = detect_query_type(query)
    top_articles = retrieve_top_k_articles(query, k=3)
    answer = generate_answer(query, top_articles, query_type)

    response = {
        "query": query,
        "question_type": query_type,
        "answer": answer,
        "evidence_list": top_articles,
      
        
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)