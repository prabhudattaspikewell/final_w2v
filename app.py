
from flask import Flask, render_template, jsonify, request

import os
import pandas as pd
import numpy
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim import corpora



app = Flask(__name__)
#api=Api(app)


df = pd.read_excel("Client_data.xlsx")

df1 = df[['Question', 'Answer']]
@app.route("/get_predict", methods=["POST"])
def answer():
    data = request.get_json()
    print(data)
    question = data['Query']
    print(question)

    print("1")

    Retrived,Answer1,score1,Retrived2,Answer2,score2 = w2v(question)

    print(2)

    return jsonify({
        "Retrived1":Retrived,
        "Answer1":Answer1,
        "Score1": str(score1),
        "Retrived2":Retrived2,
        "Answer2":Answer2,
        "Score2":str(score2)
    })
#@app.route("/new_data",methods=["POST"])
#def add():
    #new_data=request.get_json()
    #print(new_data)
    #df1 = df1.append(new_row, ignore_index=True)


df1.isna().sum()
df1.fillna("Not available")

def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence

def get_cleaned_sentences(df, stopwords=False):
    sents = df1[["Question"]]
    cleaned_sentences = []

    for index, row in df1.iterrows():
        # print(index,row)
        cleaned = clean_sentence(row["Question"], stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)

sentences = cleaned_sentences_with_stopwords
sentence_words = [[word for word in document.split()]
                  for document in sentences]



dictionary = corpora.Dictionary(sentence_words)

bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
##for sent, embedding in zip(sentences, bow_corpus):
    #sent1 = sent
    #embedding1 = embedding

#glove_model = None
# try:
#    glove_model = gensim.models.KeyedVectors.load("./glovemodel.pb")
#     print("Loaded glove model")
# except:
#     glove_model = api.load('glove-twitter-25')
#     glove_model.save("./glovemodel.pb")
#     print("Saved glove model")

v2w_model = None
try:
    v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.pb")
    print("Loaded w2v model")
except:
   v2w_model = api.load('word2vec-google-news-300')
   v2w_model.save("./w2vecmodel.pb")
   print("Saved w2v model")

w2vec_embedding_size = len(v2w_model['computer'])
# glove_embedding_size = len(glove_model['computer'])




def getWordVec(word, model):
   samp = model['computer']
   vec = [0] * len(samp)
   try:
       vec = model[word]
   except:
       vec = [0] * len(samp)
   return (vec)


def getPhraseEmbedding(phrase, embeddingmodel):
   samp = getWordVec('computer', embeddingmodel)
   vec = numpy.array([0] * len(samp));
   den = 0
   for word in phrase.split():
       print(word)
       den = den + 1
       vec = vec + numpy.array(getWordVec(word, embeddingmodel))
   # vec=vec/den
   # return (vec.tolist())
   return vec.reshape(1, -1)


def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf, sentences):
    l1 = list()
    max_sim = -1
    max_sim1 = -1
    index_sim = -1
    index_sim1=-1
    for index, faq_embedding in enumerate(sentence_embeddings):
        # sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0];
        # print(index, sim, sentences[index])

        # print(type(my_dict))
        # print(my_dict)
        # print(my_dict)
        # Keymax = max(zip(my_dict.(),my_dict.keys()))[1]
        if sim > max_sim:
            max_sim = sim
            index_sim = index
        if sim < max_sim and sim > max_sim1:
            max_sim1 = sim
            index_sim1 = index
            # inverse = [(value, key) for key, value in my_dict.items()]
            # print(max(inverse)[1])

            # print(max_sim)
        # max_key= max(my_dict.items(), key = operator.itemgetter(1))[0]
        # print(max_key)

    Retrieved= FAQdf.iloc[index_sim, 0]
    Answer= FAQdf.iloc[index_sim, 1]
    score=max_sim
    Retrived2= FAQdf.iloc[index_sim1, 0]
    Answer2=FAQdf.iloc[index_sim1, 1]
    score2=max_sim1
    return Retrieved,Answer,score,Retrived2,Answer2,score2


def w2v(question):
    sent_embeddings=[]
    for sent in cleaned_sentences:
        sent_embeddings.append(getPhraseEmbedding(sent,v2w_model))

    question_embedding=getPhraseEmbedding(question,v2w_model)
    print("Response from W2V:\n")
    return retrieveAndPrintFAQAnswer(question_embedding,sent_embeddings,df1, cleaned_sentences)


# def model1(inp):
#
#     print(3)
#
#     model = pickle.load(open('Sentence_transformer-l6.txt', 'rb'))
#     question = clean_sentence(inp, stopwords=False)
#     cleaned_sentences = get_cleaned_sentences(df, stopwords=False)
#
#     print(4)
#
#     sent_embeddings = []
#     for sent in cleaned_sentences:
#         sent_embeddings.append(model.encode([sent]))
#
#     print(5)
#
#     question_embedding = model.encode([question])
#     print("Response from model1:\n")
#     return retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, df1, cleaned_sentences)
if __name__ == '__main__':
    app.run(debug=False)
