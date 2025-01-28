# Installing all necessary libraries
import os
import pandas as pd
import csv

import xml.etree.ElementTree as ET
import requests

import nltk
from nltk.util import ngrams
nltk.download("punkt")
import string
import spacy



!!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!!pip install sentence-transformers
from sentence_transformers import SentenceTransformer

punc_without = list(string.punctuation)+["»","«"]
punc_without.remove('-')
punc_without.remove("'")
punc_all=list(string.punctuation)+["»","«"]

from IPython.display import clear_output
clear_output()

#  --------------------------Download ACTER dataset------------------------------------------
! git clone https://github.com/AylaRT/ACTER.git
clear_output()

domains = ["corp", "equi", "wind","htfl"]
langs   = ["en", "fr", "nl"]

domain   = domains[0]       # Selecting  domain
language = langs[0]         # Selecting  language

texts=[]
file_names=[]

# Extract texts  as list

folder_path = "/content/ACTER/" + language + "/" + domain + "/annotated/texts"   #unannotated_texts       annotated/texts_tokenised

file_list = os.listdir(folder_path)

for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            texts.append(text.replace("  ", " ").replace(" -","-").replace(" - ","-"))

# Extract terms as list

true_terms=[]
ann_path = '/content/ACTER/' + language + "/" + domain + "/annotated/annotations/unique_annotation_lists/" + domain + "_"+language+"_terms_nes.tsv"
with open(ann_path, 'r', newline='') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for row in reader:
        true_terms.append(row[0].lower())


true_terms_all = set([w.lower().replace("  "," ").replace("- ","-") for w in true_terms])
true_terms_mwe = set([w for w in true_terms_all if (len(w.split(" "))>1)] + [w for w in true_terms_all if len(w.split("-"))>1 ])   # True Phrase Terms
true_terms_uni = set([w for w in true_terms_all if w not in true_terms_mwe ])                                                      # True Unigrams Terms

print('True terms all: ', len(true_terms_all))
print('True terms uni: ', len(true_terms_uni))
print('True terms mwe: ', len(true_terms_mwe))


# Additional texts
# The ACTER dataset contains non-annotated texts in the domains corp, equi, wind.They are used in calculating the frequencies of candidate phrases.

text_ref=[]
if domain!="htfl":
        folder_path = "/content/ACTER/" + language + "/" + domain + "/unannotated_texts"   #unannotated_texts       annotated/texts_tokenised

        file_list = os.listdir(folder_path)

        for filename in file_list:
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    text = file.read()
                    text_ref.append(text.replace("\n"," ").replace("  ", " ").lower())

# ---------------------------------------------------------------------------------------------------------------------------


# Loading SpaCy model and stop-words depending on selected language

if language=="en":
      url = 'https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt'
      stop_words = (requests.get(url).text).split(",")
      nlp = spacy.load("en_core_web_sm")

elif language=="fr":
      !python3 -m spacy download fr_core_news_sm
      from spacy.lang.fr.examples import sentences
      url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
      stop_words = (requests.get(url).text).split("\n")
      nlp = spacy.load("fr_core_news_sm")

elif language=="nl":
      !python3 -m spacy download nl_core_news_sm
      from spacy.lang.nl.examples import sentences
      url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-nl/master/stopwords-nl.txt'
      stop_words = (requests.get(url).text).split("\n")
      nlp = spacy.load("nl_core_news_sm")

clear_output()

# Setting up the SpaCy model for word tokenization.  The rule is set: NOT split words with hyphens into separate tokens.

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from operator import itemgetter

# Modify tokenizer infix patterns
infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # Commented out regex that splits on hyphens between letters:
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

clear_output()


# POS tag templates for phrase extraction

# Selecting Parts of Speech Templates by Language
if language=="en":
  pos_tag_patterns=  [[["PROPN","NOUN"],"*"],
                     ["ADJ",'*', ["PROPN","NOUN"], '*'],
                     ["ADJ","*"],
                     ['VERB', 'ADJ', ["PROPN","NOUN"],'*'],
                     [["PROPN","NOUN"],'*','ADJ','*',["PROPN","NOUN"],'*'],
                     ['ADJ','VERB',["PROPN","NOUN"], '*'],
                     ['VERB','*',["PROPN","NOUN"],'*'],
                     [["PROPN","NOUN"], '*','ADJ','*',["PROPN","NOUN"], '*'],
                     ["ADV","*","ADJ","*"],
                     [["PROPN","NOUN"],'ADP',["PROPN","NOUN"]],
                     [["ADJ","PROPN","NOUN"],"*","PART",["PROPN","NOUN"],"*"],
                     [['VERB',"ADV","X"]],
                     [["ADJ","PROPN","NOUN"],"*", "ADP",["PROPN","NOUN"],"*"]
                      ]

elif language=="fr":
  pos_tag_patterns=[[["NOUN","ADJ","PROPN"],"*"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"],
                    [["NOUN","ADJ"],"*","ADP","DET",["NOUN","ADJ"],"*"],
                    ["NOUN","VERB"],
                    ["VERB","ADJ"],
                    ["ADJ","VERB"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"]]


elif language=="nl":
   pos_tag_patterns=[[["NOUN","ADJ","PROPN","SYM"],"*"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ","CCONJ"],"*"],
                    ["NOUN","ADP","NOUN","ADP","NOUN"],
                    ["VERB","ADP","NOUN"],
                    ["VERB","ADP","DET","NOUN"],
                    ["VERB","NOUN"]]



# --------------------Loading a model for extracting multi-word expressions------------------------
!git clone https://github.com/term-extraction-project/multi_word_expressions.git

import sys
sys.path.append('/content/multi_word_expressions/extractors')

phrases=[]
# Selecting a Phrase Extraction Model Based on Language

if language=="en":
    from english import EnglishPhraseExtractor
    for ind, text in enumerate(texts):

        all_t=" .".join(texts[:ind] + texts[ind+1:]) + " .".join(text_ref)    # All texts except the target one are needed to extract frequency-based candidate phrase filtering
        extractor = EnglishPhraseExtractor(text=text,                         # The target text is submitted in string format.
                                    stop_words=stop_words,                    # Loading stop words 
                                    cohision_filter=True,                     # Enable the use of the cohesive filter. It is based on frequency calculation, and this filter requires additional_text
                                    additional_text=all_t,                    # Use additional texts in string format
                                    list_seq=pos_tag_patterns,                # Loading part-of-speech templates
                                    f_raw_sc=9,                               # Raw frequency threshold, not related to cohision_filter
                                    f_req_sc=3)                               # Rectified frequency threshold, not related to cohision_filter
        candidatese = extractor.extract_phrases()                             # Extracting phrases
        phrases += candidatese
    clear_output()
    print(f'Number of extracted phrases: {len(set(phrases))}')

elif language=="fr":
     from french import FrenchPhraseExtractor
     for ind, text in enumerate(texts):

        all_t=" .".join(texts[:ind] + texts[ind+1:]) + " .".join(text_ref)

        extractor = FrenchPhraseExtractor(text=text,
                                    stop_words=stop_words,
                                    cohision_filter=True,
                                    additional_text=all_t,
                                    list_seq=pos_tag_patterns,
                                    f_raw_sc=9,
                                    f_req_sc=3)
        candidatese = extractor.extract_phrases()
        phrases += candidatese
     clear_output()
     print(f'Number of extracted phrases: {len(set(phrases))}')

elif language=="nl":
     from dutch import DutchPhraseExtractor
     for ind, text in enumerate(texts):
        all_t=" .".join(texts[:ind] + texts[ind+1:]) + " .".join(text_ref)
        extractor = DutchPhraseExtractor(text=text,
                                    stop_words=stop_words,
                                    cohision_filter=True,
                                    additional_text=all_t,
                                    list_seq=pos_tag_patterns,
                                    f_raw_sc=9,
                                    f_req_sc=3)
        candidatese = extractor.extract_phrases()
        phrases += candidatese
     clear_output()
     print(f'Number of extracted phrases: {len(set(phrases))}')


# -------------Extracting sentences, unigrams and abbreviations--------------------
unigrams=[]
abb=[]
sents=[]

for tex in texts:
        text_token=nlp(tex)

        # Extract sentences, convert to lowercase, and add to 'sents'
        se=[sent.text.lower() for sent in text_token.sents if len(sent.text) > 0]
        sents+=se

        # Extract abbreviations
        for_abb=[i.text for i in text_token]
        # Filter abb tokens: remove those in stop words, punctuation, numeric tokens, tokens consisting only of punctuation and numbers 
        for_abb=[w for w in for_abb if w.lower() not in stop_words and len(set(w).intersection(set(list("0123456789")+list(punc_all))))<len(set(w))]
        for_abb=[w  for w in for_abb if w not in punc_all and len(set(w).intersection(set(punc_all)))==0]
        for_abb=[i for i in for_abb if sum(1 for char in i if char.isupper())>1 and len(i)<30]   # Retain only tokens with more than one uppercase letter and a maximum length of 30

        abb+=for_abb


        # Extract Unigrams
        # Extarct tokens with parts of speech (proper nouns, nouns, adjectives), convert to lowercase
        text_token=[i.text.lower() for i in text_token if i.pos_ in ["PROPN","NOUN","ADJ"]]

        # Filter unigram-tokens: remove stop words, numeric/punctuation tokens, tokens consisting only of punctuation and numbers, tokens with punctuation at the start/end or within
        text_token=[w for w in text_token if w.lower() not in stop_words and len(set(w).intersection(set(list("0123456789")+list(punc_all))))<len(set(w))]
        text_token=[w for w in text_token if w not in punc_all and len(set(w).intersection(set(punc_without)))==0 and w[0] not in punc_all and w[-1] not in punc_all]
        unigrams+=text_token

# Lowercase all abbreviations
abb_lower = set([i.lower() for i in abb])

# The rule was set to not split words with hyphens, so some multi-word candidates could be extracted with unigrams. 
# This rule will extract such candidates from the unigram list.

phrase_def = [i for i in set(unigrams) if len(i.split("-"))>1] + [i for i in set(unigrams) if len(i.split("'"))>1]
unigrams = set(unigrams) - set(phrase_def)
phrases = set(phrases + phrase_def)


# -------------------------Evaluation of candidate extraction efficiency-----------------------------------------------
def calculate_metrics(true_terms, extracted_terms):
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

# Checking the quality of extraction of unigram candidates, phrases and general, excluding abbreviations.
precision, recall, f1_score=calculate_metrics(true_terms_uni, unigrams)
print("Single word terms extraction results")
print("N: ", len(unigrams))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score=calculate_metrics(true_terms_mwe, phrases)
print("Phrases terms extraction results")
print("N: ", len(phrases))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score=calculate_metrics(true_terms_all, unigrams.union(phrases))
print("Total extraction results")
print("N: ", len(unigrams.union(phrases)))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))



# ------------------------------- Semantic filter based on Topic Score---------------------------------------

# Loading a model to convert sentences, phrases and unigrams into contextual embeddings

if language=="en":
      model_encode = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
else:
      model_encode = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

clear_output()

sents_en=[[i, model_encode.encode(i, normalize_embeddings=True)] for i in sents] # Receiving sentence embeddings
# Receiving unigrams embeddings and obtaining cosine similarity with a sentence containing the target unigram
cos_uni=[]
for i in unigrams:
  i_en= model_encode.encode(i, normalize_embeddings=True)
  for s in sents_en:
     if i.lower() in s[0].lower():
       s_en=s[1]
       topic_score=model_encode.similarity(i_en, s_en).tolist()[0][0]
       cos_uni.append([i,topic_score])

# Receiving phrase embeddings and obtaining cosine similarity with a sentence containing the target phrase
cos_mwe=[]
for i in phrases:
  i_en= model_encode.encode(i, normalize_embeddings=True)
  for s in sents_en:
     if i.lower() in s[0].lower():
       s_en=s[1]
       topic_score=model_encode.similarity(i_en, s_en).tolist()[0][0]
       cos_mwe.append([i,topic_score])

#------------------ Evaluating effectiveness of Semantic Filtering----------------------
# Setting the TopicScore threshold and assessing the quality of semantic filtering for unigrams, phrases and general without abbreviations
uni = set([i[0] for i in cos_uni if i[1]>0.4])
mwe = set([i[0] for i in cos_mwe if i[1]>0.4])

precision, recall, f1_score=calculate_metrics(true_terms_uni, uni)
print("Single word terms extraction results")
print("N: ", len(uni))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score=calculate_metrics(true_terms_mwe, mwe)
print("Phrases terms extraction results")
print("N: ", len(mwe))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score=calculate_metrics(true_terms_all, uni.union(mwe))
print("Total extraction results")
print("N: ", len(uni.union(mwe)))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))



# ----------------------------Evaluating the effectiveness of adding Abbreviations------------------

precision, recall, f1_score=calculate_metrics(true_terms_all, abb_lower)
print("Precision extraction by using ABB rule")
print("N: ", len(abb_lower))
print("Precision: ", round(precision*100,2))

# Evaluating the performance of term extraction with the addition of abbreviations
abb_mwe = set([ i for i in abb_lower if len(i.split("-"))>1 ] + [ i for i in abb_lower if len(i.split("'"))>1 ])
abb_uni = abb_lower - abb_mwe

precision, recall, f1_score=calculate_metrics(true_terms_uni, uni.union(abb_uni))
print("Single word terms extraction results")
print("N: ", len(uni.union(abb_uni)))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score=calculate_metrics(true_terms_mwe, mwe.union(abb_mwe))
print("Phrases terms extraction results")
print("N: ", len(mwe.union(abb_mwe)))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score=calculate_metrics(true_terms_all, set(list(uni)+list(mwe)+list(abb_lower)))
print("Total extraction results")
print("N: ", len(set(list(uni)+list(mwe)+list(abb_lower))))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))

# --------------------------------------NE extraction------------------------------------------------------------------
# Extraction of PROPN sequences. Also takes into account that the middle may contain ADP

propn_sequence=[]

for text in texts:
  doc = nlp(text)
  temp = ""
  temp_2 = ''
  check = False

  for i in doc:
    if i.pos_ == "PROPN":
      if len(temp_2) > 0:
        check = True
        temp = temp_2
      temp = (temp + " " + i.text).strip()
    elif i.pos_ == "ADP" and len(temp) > 0:
       temp_2 = temp + " " + i.text
    else:
      if len(temp) > 0:
         propn_sequence.append(temp)
      temp = ""
      temp_2 = ""
  if len(temp) > 0:
         propn_sequence.append(temp)

propn_sequence_lower = set([i.lower() for i in set(propn_sequence) if len(set(i.lower()).intersection(set(punc_all+list(string.digits))))==0])

# Evaluating the accuracy of extracting relevant words using this rule

precision, recall, f1_score=calculate_metrics(true_terms_all, propn_sequence_lower)
print("Precision extraction by using NE rule")
print("N: ", len(propn_sequence_lower))
print("Precision: ", round(precision*100,2))

# ------------------------------------ Overall extraction score using T-Extractor --------------------------------------------
# Separation of extracted propn sequences into single-word and multi-word. Necessary when evaluating the efficiency of unigram and phrase extraction.

mwe_propn_sequence_lower = set([i for i in propn_sequence_lower if len(i.split(" "))>1] + [i for i in propn_sequence_lower if len(i.split("-"))>1])
uni_propn_sequence_lower = propn_sequence_lower - mwe_propn_sequence_lower

# Final evaluation of term extraction taking into account abbreviations and named entities.

total_unigrams = set(list(uni) + list(abb_uni) + list(uni_propn_sequence_lower))
total_phrases = set(list(mwe)+list(abb_mwe)+list(mwe_propn_sequence_lower))
total_terms = set(list(uni)+list(mwe)+list(abb_lower)+list(propn_sequence_lower))

precision, recall, f1_score = calculate_metrics(true_terms_uni, total_unigrams)
print("Single word terms extraction results")
print("N: ", len(total_unigrams))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score = calculate_metrics(true_terms_mwe, total_phrases)
print("Phrases terms extraction results")
print("N: ", len(total_phrases))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))
print()

precision, recall, f1_score = calculate_metrics(true_terms_all, total_terms)
print("Total extraction results")
print("N: ", len(total_terms))
print("Precision: ", round(precision*100,2))
print("Recall: ", round(recall*100,2))
print("F1 Score: ", round(f1_score*100,2))

# -------------------------------- Output extracted terms and named entities ---------------------------------------------
print("Extracted terms and NEs:")
print(len(total_terms))
print(sorted(total_terms))
print()

print("True positive extracted terms and NEs:")
print(len(total_terms.intersection(true_terms_all)))
print(sorted(total_terms.intersection(true_terms_all)))
print()

print("False positive extracted terms and NEs:")
print(len(total_terms - true_terms_all))
print(sorted(total_terms - true_terms_all))
print()







































