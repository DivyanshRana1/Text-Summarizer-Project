import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def read_text():
    text = input("Paste the text to summarize:\n")
    return text

def preprocess_sentences(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed = []

    for sentence in sentences:
        words = [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
        words = [w for w in words if w not in stop_words]
        processed.append(words)

    return sentences, processed

def sentence_similarity(sent1, sent2):
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - np.linalg.norm(np.array(vector1) - np.array(vector2))

def build_similarity_matrix(sentences):
    n = len(sentences)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    return sim_matrix

def summarize(text, top_n=3):
    sentences, processed = preprocess_sentences(text)
    sim_matrix = build_similarity_matrix(processed)
    scores = nx.pagerank(nx.from_numpy_array(sim_matrix))
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary = " ".join([s for _, s in ranked[:top_n]])
    return summary

if __name__ == "__main__":
    text = read_text()
    summary = summarize(text)
    print("\nSummary:\n", summary)
