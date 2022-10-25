from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim import corpora

# app definition
app = Flask(__name__)
CORS(app)

df = pd.read_excel("Client_data.xlsx")
df1 = df[['Question', 'Answer']]


@app.route("/get_predict", methods=["POST"])
@cross_origin()
def answer():
    data = request.get_json()
    print(data)
    question = data['Query']
    print(question)

    retrieved1, answer1, score1, retrieved2, answer2, score2 = w2v(question)

    return jsonify({
        "retrieved1": retrieved1,
        "answer1": answer1,
        "Score1": str(score1),
        "retrieved2": retrieved2,
        "answer2": answer2,
        "Score2": str(score2)
    })


df1.isna().sum()
df1.fillna("Not available")


def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence


def get_cleaned_sentences(stopwords=False):
    clear_sentences = []

    for index, row in df1.iterrows():
        # print(index,row)
        cleaned = clean_sentence(row["Question"], stopwords)
        clear_sentences.append(cleaned)
    return clear_sentences


# passed 2 arguments, a df and a bool, what to do here?
cleaned_sentences = get_cleaned_sentences(stopwords=True)
cleaned_sentences_with_stopwords = get_cleaned_sentences(stopwords=False)

sentences = cleaned_sentences_with_stopwords
sentence_words = [[word for word in document.split()] for document in sentences]

dictionary = corpora.Dictionary(sentence_words)

bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
v2w_model = None
try:
    v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.pb")
    print("Loaded w2v model")
except Exception:
    v2w_model = api.load('word2vec-google-news-300')
    v2w_model.save("./w2vecmodel.pb")
    print("Saved w2v model")

w2vec_embedding_size = len(v2w_model['computer'])


def get_word_vec(word, model):
    samp = model['computer']
    try:
        vec = model[word]
    except Exception:
        vec = [0] * len(samp)
    return vec


def get_phrase_embedding(phrase, embedding_model):
    samp = get_word_vec('computer', embedding_model)
    vec = numpy.array([0] * len(samp))
    den = 0
    for word in phrase.split():
        # print(word)
        den = den + 1
        vec = vec + numpy.array(get_word_vec(word, embedding_model))
    return vec.reshape(1, -1)


def retrieve_and_print_faq_answer(question_embedding, sentence_embeddings, faq_df):
    max_sim = -1
    max_sim1 = -1
    index_sim = -1
    index_sim1 = -1
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]

        if sim > max_sim:
            max_sim = sim
            index_sim = index
        if max_sim > sim > max_sim1:
            max_sim1 = sim
            index_sim1 = index

    retrieved = faq_df.iloc[index_sim, 0]
    answer1 = faq_df.iloc[index_sim, 1]
    score = max_sim
    retrieved2 = faq_df.iloc[index_sim1, 0]
    answer2 = faq_df.iloc[index_sim1, 1]
    score2 = max_sim1
    return retrieved, answer1, score, retrieved2, answer2, score2


def w2v(question):
    sent_embeddings = []
    for sent in cleaned_sentences:
        sent_embeddings.append(get_phrase_embedding(sent, v2w_model))

    question_embedding = get_phrase_embedding(question, v2w_model)
    print("Response from W2V:\n")
    return retrieve_and_print_faq_answer(question_embedding, sent_embeddings, df1)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
    # app.run(debug=True)
