# T-Extractor

T-Extractor is an unsupervised annotation tool designed to extract terms and named entities using a combination of rule-based, statistical, and semantic analysis methods. Currently, T-Extractor supports English, French, and Dutch languages.


# Installation and Usage

This guide provides instructions for installing and using T-Extractor in Google Colab. You can also access the code via the following [link](https://colab.research.google.com/drive/1eYdumGQ8bA3MUd-MCGIdBoNnX3Nm_S_N?usp=sharing).

Before using T-Extractor, ensure that the required libraries and models are installed.

## Installation

First, install **T-Extractor**:

```bash

!git clone https://github.com/term-extraction-project/T-Extractor.git
import sys
sys.path.append('/content/T-Extractor') # Path to the python file where the T-Extractor code is located

```

Next, clone the Phrase Extractor repository and install the necessary dependencies:

```bash
# Installing Phrase Extractor
!git clone https://github.com/term-extraction-project/multi_word_expressions.git
%cd multi_word_expressions

# Download sentence-transformers for encoding sentences and candidates
!!pip install sentence-transformers

# Download the required SpaCy model for French or Dutch
!python3 -m spacy download fr_core_news_sm    # for French
!python3 -m spacy download nl_core_news_sm    # for Dutch

```


## Usage

To use T-Extractor, import the module and process the text as shown below:

```bash
from unsupervised_term_extractor import T_Extractor  # Connecting T_Extractor

# The input is a list containing many texts or one text in string format.
text_en = ["""
T-Extractor is an unsupervised annotation tool designed to extract terms and named entities using a combination of rule-based, statistical, and semantic analysis methods. Currently, T-Extractor supports English, French, and Dutch languages.
""",
"""
T-Extractor was tested on the ACTER (three languages and four domains) and ACL RD-TEC 2.0 datasets, where the average F1 score was about 40% on English, outperform-ing some supervised methods. """
]

templetes = [
              "PROPN",
              "NOUN",
              "ADJ",
              [["PROPN","NOUN"],"*"],
              ["ADJ",'*', ["PROPN","NOUN"], '*']
             ]

# Model setup
extractor = T_Extractor(
                           text = text_en,                 #  Texts from which terms need to be extracted
                           lang = "en",                    #  Choice of language from English(en), French(fr) and Dutch(nl). English is the default
                           additional_text = "",           #  If there is text of the same domain as the target. Needed to increase the frequency of phrases. 
                           cohision_filter = True,         #  Enabled by default. Filtering extracted phrases using frequencies.
                           f_raw = 9,                      #  Raw frequency threshold
                           f_req = 3,                      #  Rectified frequency threshold 
                           pos_tag_patterns = templetes,   #  By default, they are embedded in the model. You can apply your own templates.
                           topic_score = 0.4,              #  Topic score threshold
                           abb_extraction = True,          #  Use or not additional rule for extracting abbreviations
                           propn_seq_extraction = True     #  Use or not additional rule for extracting sequences of proper nouns (PRON)
                       )

candidatese = extractor.term_extraction()

print(candidatese)

```


# Evaluation

T-Extractor was tested on ACTER and ACL RD-TEC 2.0 datasets. The test code is available on [Гугл Коллаб](https://colab.research.google.com/drive/1LgGsv5FawMZOVrFhhpIpkrqbgx1q4nu3?usp=sharing). 


