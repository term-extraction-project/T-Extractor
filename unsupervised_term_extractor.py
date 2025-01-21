import nltk
nltk.download("punkt")
import string
import spacy

import requests


from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from operator import itemgetter


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sentence_transformers import SentenceTransformer


punc_without = list(string.punctuation)+["»","«"]
punc_without.remove('-')
punc_without.remove("'")
punc_all=list(string.punctuation)+["»","«"]


def define_pos_pattern(language):
    pos_tag_patterns=[]
    if language=="en":
      pos_tag_patterns=  ["PROPN","NOUN","ADJ",
                          [["PROPN","NOUN"],"*"],
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
      pos_tag_patterns=["PROPN","NOUN","ADJ",
                        [["NOUN","ADJ","PROPN"],"*"],
                        [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"],
                        [["NOUN","ADJ"],"*","ADP","DET",["NOUN","ADJ"],"*"],
                        ["NOUN","VERB"],
                        ["VERB","ADJ"],
                        ["ADJ","VERB"],
                        [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"],
                        [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"]]


    elif language=="nl":
      pos_tag_patterns=["PROPN","NOUN","ADJ",
                        [["NOUN","ADJ","PROPN","SYM"],"*"],
                        [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ","CCONJ"],"*"],
                        ["NOUN","ADP","NOUN","ADP","NOUN"],
                        ["VERB","ADP","NOUN"],
                        ["VERB","ADP","DET","NOUN"],
                        ["VERB","NOUN"]]
    return pos_tag_patterns



class T_Extractor:
    def __init__(self, text, lang="en", additional_text="", cohision_filter=True, f_raw=9, f_req=3, pos_tag_patterns=[], topic_score=0.4, abb_extraction=True, propn_seq_extraction=True ):
        self.text = text            # текст в оригинальном регистре
        self.lang=lang
        self.additional_text=additional_text   # если есть дополнительный текст, используется для вычисления частот, из него термины НЕ извлекаються
        self.cohision_filter=cohision_filter     #  Включить или отключить когезионный фильтр
        self.f_raw=f_raw     # порог выпрямленной частоты
        self.f_req=f_req       # порог сырой частоты
        self.pos_tag_patterns=pos_tag_patterns
        self.topic_score=topic_score
        self.abb_extraction=abb_extraction
        self.propn_seq_extraction=propn_seq_extraction

    def term_extraction(self):

        pos_tag_patterns=self.pos_tag_patterns
        if len(pos_tag_patterns)==0:
            pos_tag_patterns=define_pos_pattern(self.lang)

        if len(pos_tag_patterns)==0:
            print(f'For the language "{self.lang}", T-Extractor cannot extract terms from text yet.\n'
                  'The annotator can extract terms from English (en), French (fr), and Dutch (nl) texts only.\n'
                  'You can use the open-source code of T-Extractor and adapt it for yourself.')
            return 0


        uni_pos_patterns = [pattern for pattern in pos_tag_patterns if isinstance(pattern, str)]
        mwe_pos_patterns = [pattern for pattern in pos_tag_patterns if isinstance(pattern, list)]


        texts = [text.replace("  ", " ").replace(" -","-").replace(" - ","-") for text in self.text]

        if self.lang=="en":
              url = 'https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt'
              stop_words = (requests.get(url).text).split(",")
              nlp = spacy.load("en_core_web_sm")
              from multi_word_expressions.extractors.english import EnglishPhraseExtractor
              model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)


        elif self.lang=="fr":
              from spacy.lang.fr.examples import sentences
              url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
              stop_words = (requests.get(url).text).split("\n")
              nlp = spacy.load("fr_core_news_sm")
              from multi_word_expressions.extractors.french import FrenchPhraseExtractor
              model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

        elif self.lang=="nl":
              from spacy.lang.nl.examples import sentences
              url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-nl/master/stopwords-nl.txt'
              stop_words = (requests.get(url).text).split("\n")
              nlp = spacy.load("nl_core_news_sm")
              from multi_word_expressions.extractors.dutch import DutchPhraseExtractor
              model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


        phrases=[]
        for ind, text in enumerate(texts):

            all_texts=" .".join(texts[:ind] + texts[ind+1:]) + " .".join(self.additional_text)
            if self.lang=="en":
                    extractor = EnglishPhraseExtractor(text=text,                   
                                                       stop_words=stop_words,
                                                       cohision_filter=self.cohision_filter,
                                                       additional_text=all_texts,
                                                       list_seq=mwe_pos_patterns,
                                                       f_raw_sc= self.f_raw,
                                                       f_req_sc=self.f_req)
                    candidatese = extractor.extract_phrases()
                    phrases += candidatese
            elif self.lang=="fr":
                    extractor = FrenchPhraseExtractor(text=text,                    
                                                       stop_words=stop_words,
                                                       cohision_filter=self.cohision_filter,
                                                       additional_text=all_texts,
                                                       list_seq=mwe_pos_patterns,
                                                       f_raw_sc= self.f_raw,
                                                       f_req_sc=self.f_req)
                    candidatese = extractor.extract_phrases()
                    phrases += candidatese
            elif self.lang=="nl": 
                    extractor = DutchPhraseExtractor(text=text,                    
                                                       stop_words=stop_words,
                                                       cohision_filter=self.cohision_filter,
                                                       additional_text=all_texts,
                                                       list_seq=mwe_pos_patterns,
                                                       f_raw_sc= self.f_raw,
                                                       f_req_sc=self.f_req)
                    candidatese = extractor.extract_phrases()
                    phrases += candidatese

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

        unigrams=[]
        abb=[]
        sents=[]
        for tex in texts:
                text_token=nlp(tex)
                se=[sent.text.lower() for sent in text_token.sents if len(sent.text) > 0]
                sents+=se
                if self.abb_extraction==True:
                    for_abb=[i.text for i in text_token]
                    for_abb=[w for w in for_abb if w.lower() not in stop_words and len(set(w).intersection(set(list("0123456789")+list(punc_all))))<len(set(w))]
                    for_abb=[w  for w in for_abb if w not in punc_all and len(set(w).intersection(set(punc_all)))==0]
                    for_abb=[i for i in for_abb if sum(1 for char in i if char.isupper())>1 and len(i)<30]
                    abb+=for_abb

                text_token=[i.text.lower() for i in text_token if i.pos_ in uni_pos_patterns]
                text_token=[w for w in text_token if w.lower() not in stop_words and len(set(w).intersection(set(list("0123456789")+list(punc_all))))<len(set(w))]
                text_token=[w for w in text_token if w not in punc_all and len(set(w).intersection(set(punc_without)))==0 and w[0] not in punc_all and w[-1] not in punc_all]
                unigrams+=text_token
        abb_lower=[i.lower() for i in abb]

        phrase_def=[i for i in set(unigrams) if len(i.split("-"))>1]+[i for i in set(unigrams) if len(i.split("'"))>1]

        sents_encode=[[i,model.encode(i, normalize_embeddings=True)] for i in sents]

        cos_uni=[]
        for i in set(unigrams)-set(phrase_def):
          i_en= model.encode(i, normalize_embeddings=True)
          for s in sents_encode:
            if i.lower() in s[0].lower():
              s_en=s[1]
              topic_score=model.similarity(i_en, s_en).tolist()[0][0]
              cos_uni.append([i,topic_score])

        cos_mwe=[]
        for i in set(phrases+phrase_def):
          i_en= model.encode(i, normalize_embeddings=True)
          for s in sents_encode:
            if i.lower() in s[0].lower():
              s_en=s[1]
              topic_score=model.similarity(i_en, s_en).tolist()[0][0]
              cos_mwe.append([i,topic_score])

        uni=[i[0] for i in cos_uni if i[1]>self.topic_score]
        mwe=[i[0] for i in cos_mwe if i[1]>self.topic_score]

        propn_sequence=[]
        if self.propn_seq_extraction==True:
              for text in texts:
                doc=nlp(text)
                temp=""
                temp_2=''
                check=False
                for i in doc:
                  if i.pos_=="PROPN":

                    if len(temp_2)>0:
                      check=True
                      temp=temp_2
                    temp=(temp+" "+i.text).strip()
                  elif i.pos_=="ADP" and len(temp)>0:
                    temp_2=temp+" "+i.text
                  else:
                    if len(temp)>0:
                      propn_sequence.append(temp)
                    temp=""
                    temp_2=""
                if len(temp)>0:
                      propn_sequence.append(temp)

              propn_sequence=[i.lower() for i in set(propn_sequence) if len(set(i.lower()).intersection(set(punc_all+list(string.digits))))==0]

        extracted_terms=set(uni + mwe + abb_lower + propn_sequence)

        return extracted_terms
