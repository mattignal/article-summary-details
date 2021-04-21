# article-summary-details [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/mattignal/article-summary-details/main/app.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattignal/article-summary-details/blob/main/Article_Summary_Details.ipynb)

Creates abstract, key ideas, details, and a summary. This repo contains two options:

[Streamlit](https://share.streamlit.io/mattignal/article-summary-details/main/app.py): User-friendly interface which uses the Distilbart 6-6 model. This is the faster, lighter, option. More importantly, it is small enough to run successfully on Streamlit!

[Colab](https://colab.research.google.com/github/mattignal/article-summary-details/blob/main/Article_Summary_Details.ipynb): Python notebook interface which uses full bart-large-cnn model. Better summarizations.

#### Progress (bold indicates if complete)

1. **Get and format articles automatically**
2. **Create abstract generator**
3. **Create key ideas generator**
4. **Create details generator**
5. **Create summary generator**
6. **Build app in Streamlit** (see badge at top)

#### Article Summary Details (BART)

## Models and Tokenizers
BART, or Bidirectional and Auto-Regressive Transformers, will be used for this task as it performs well for summarization tasks. According to the docs:

> BART uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). 

> The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.

> BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

Here we will use BART-CNN, which has been fine-tuned on the CNN article/summarization datatest.


<img src="https://github.com/mattignal/article-summary-details/blob/main/bart_app.png">

<img src="https://github.com/mattignal/article-summary-details/blob/main/Abstract_details.png">
