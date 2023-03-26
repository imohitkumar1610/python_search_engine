import pandas as pd
import numpy as np

import operator
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

stop = list(stopwords.words('english'))

import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

import re
from functools import reduce
import torch

text = "at 7am only the boy has to wake and see wheater this is monday or not. Position the cursor in the document where you want to generate random text."

doc = nlp(text)
if doc.ents:
  for ent in doc.ents:
    print("{}: {}".format(ent.text, ent.label_))

    for chunk in doc.noun_chunks:
  print(chunk.text)

  str1 = "magnetism"
str2 = "MagNetism"

Ratio=fuzz.ratio(str1,str2)
print(Ratio)

stop = []

f = open(r'/content/STOPWORDS.txt', 'r')
for l in f.readlines():
    stop.append(l.replace('\n', ''))

additional_stop_words = ['video', 'videos']
stop += additional_stop_words


# Snippet 4

import pandas as pd

df = pd.read_csv(r"/content/video embedding.csv")
df.head()

# Snippet 5

def spacy_nounphrase(text):
    ent = []
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        ent.append(str(chunk.text))
    return ent

def ner_spacy(text):
    entities = []
    doc = nlp(text)
    if doc.ents:
        for ent in doc.ents:
            entities.append(ent.text)
    return entities

def extract_entities(text, stop_words):
    ent1 = spacy_nounphrase(text)
    ent2 = ner_spacy(text)
    ents = list(set(ent1+ent2))
    entities_ = [x.lower() for x in ents]
    entities = [word for word in entities_ if word not in stop_words]
    return entities


# Snippet 6

def search_phrase_fuzzy(df, entities):
    video_scores = []
    for index, rows in df.iterrows():
        title = extract_entities(rows[0], stop)
        body = extract_entities(rows[2], stop)
        title_score = 0
        body_score = 0
        for ent in entities:
            match_title = process.extract(ent,title,scorer=fuzz.partial_ratio)
            match_title = [sub for sub in match_title if sub[1] > 90] 
            title_score += len(match_title)
            match_body = process.extract(ent,body,scorer=fuzz.partial_ratio)
            match_body = [sub for sub in match_body if sub[1] > 90] 
            body_score += len(match_body)
        
        score = 0.7*title_score + 0.3*body_score
        video_scores.append(score)
    
    df['Video Scores'] = video_scores
    return df

# Snippet 7

def fetch_videos(text, df, stop):
    tokens = extract_entities(text, stop)
    print("\n\n-------------------------------------------------------------------------\n\nEntities extracted from the query:")
    print("{}\n\n-------------------------------------------------------------------------\n\n".format(tokens))
    df = search_phrase_fuzzy(df, tokens)
    
    dy = df.sort_values(by=['Video Scores'], ascending=False).iloc[0:10]
    
    return dy

text = "chess"
fetch_videos(text, df, stop)

text="search for cold war"
fetch_videos(text,df,stop)

# Snippet 8

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/sentence-t5-base')

txt = "three years later, the coffin was still full of jelio"

sen = ["the fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.","the person box was packed with jelly many dozen of months later"]

txt_embeddings = model.encode(txt)
sen_embeddings = model.encode(sen)


# Snippet 9

from rpunct import RestorePuncts

rpunct = RestorePuncts()


text = "and there is one man standing at the backdoor waiting for the session to end and he will get the food"
rpunct.punctuate(text, lang = 'en')


# Snippet 10

def search_score_t5(df, embeddings):
    video_scores = []
    for index, rows in df.iterrows():
        title = rpunct.punctuate(rows[0], lang='en')
        title = title.strip(".").split(".")
        title_embed = model.encode(title)
        body = rpunct.punctuate(rows[2], lang='en')
        body = body.strip(".").split(".")
        body_embed = model.encode(body)

        title_scores = util.pytorch_cos_sim(embeddings, title_embed)
        body_scores = util.pytorch_cos_sim(embeddings, body_embed) 
        title_score = torch.max(title_scores)
        body_score = torch.max(body_scores)

        score = 0.5*title_score + 0.5*body_score
        video_scores.append(score)
    
    df['Video Scores'] = video_scores
    return df

# Snippet 11

def fetch_videos(text, df):

    embeddings = model.encode(text)
    df = search_score_t5(df, embeddings)
    
    dy = df.sort_values(by=['Video Scores'], ascending=False).iloc[0:10]
    
    return dy

text="doctor"
fetch_videos(text,df)