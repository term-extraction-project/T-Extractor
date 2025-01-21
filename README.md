# T-Extractor

T-Extractor is an unsupervised annotation tool designed to extract terms and named entities using a combination of rule-based, statistical, and semantic analysis methods. Currently, T-Extractor supports English, French, and Dutch languages.


# Installation and Usage

This guide provides instructions for installing and using T-Extractor in Google Colab. You can also access the code via the following [link](https://colab.research.google.com/drive/1eYdumGQ8bA3MUd-MCGIdBoNnX3Nm_S_N?usp=sharing).

Before using T-Extractor, ensure that the required libraries and models are installed.

## Installation

First, clone the Phrase Extractor repository and install the necessary dependencies:

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

Next, install **T-Extractor**:

```bash

!git clone https://github.com/term-extraction-project/T-Extractor.git
import sys
sys.path.append('/content/T-Extractor') # Path to the python file where the T-Extractor code is located

```

## Usage

To use T-Extractor, import the module and process the text as shown below:

```bash
from unsupervised_term_extractor import T_Extractor

text_en = ["""
T-Extractor is an unsupervised annotator that extracts terms and named entities based on rules, statistical and semantic analysis.
"""]

extractor = T_Extractor(text=text, lang="fr" )
candidatese = extractor.term_extraction()

print(candidatese)

```


# Оценка эффективности

T-Extractor был протестирован на набоах данных ACTER и ACL RD-TEC 2.0. Код тестирования доступен на [Гугл Коллаб] (https://colab.research.google.com/drive/1LgGsv5FawMZOVrFhhpIpkrqbgx1q4nu3?usp=sharing). 


# Результаты тестирования подхода на на набоах данных ACTER и ACL RD-TEC 2.0

ACL-RD-TEC 2.0 (English)

| Annotator  | Precision (P) | Recall (R) | F1-score (F1) |
|------------|--------------|------------|--------------|
| Annotator 1| 35.01         | 61.18      | 44.54        |
| Annotator 2| 33.77         | 61.77      | 43.67        |


Acter Dataset

English
| Domain | Precision (P) | Recall (R) | F1-score (F1) |
|--------|--------------|------------|--------------|
| Corp   | 31.47         | 55.33      | 40.12        |
| Equi   | 41.56         | 58.16      | 48.48        |
| Wind   | 28.60         | 58.30      | 38.37        |
| HTFL   | 43.68         | 48.66      | 46.04        |

French
| Domain | Precision (P) | Recall (R) | F1-score (F1) |
|--------|--------------|------------|--------------|
| Corp   | 25.43         | 53.44      | 34.46        |
| Equi   | 22.70         | 51.57      | 31.53        |
| Wind   | 17.42         | 63.02      | 27.30        |
| HTFL   | 42.51         | 50.84      | 46.31        |

Dutch
| Domain | Precision (P) | Recall (R) | F1-score (F1) |
|--------|--------------|------------|--------------|
| Corp   | 29.03         | 63.78      | 39.90        |
| Equi   | 35.58         | 65.93      | 46.22        |
| Wind   | 20.83         | 68.54      | 31.95        |
| HTFL   | 40.40         | 62.24      | 49.00        


О подходе
