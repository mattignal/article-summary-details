# article-summary-details
(WIP) [Generates abstract, key ideas, and related details from articles](https://colab.research.google.com/github/mattignal/article-summary-details/blob/main/Article_Summary_Details.ipynb)

#### Progress (bold indicates if complete)

1. **Get and format articles automatically**
2. **Create abstract generator**
3. **Create key ideas generator**
4. **Create details generator**
5. **Create summary generator**
6. Build app

#### Article Summary Details (BART)

## Models and Tokenizers
BART, or Bidirectional and Auto-Regressive Transformers, will be used for this task as it performs well for summarization tasks. According to the docs:

> BART uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). 

> The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.

> BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

Here we will use BART-CNN, which has been fine-tuned on the CNN article/summarization datatest.

<img src="https://github.com/mattignal/article-summary-details/blob/main/bart_app.png">

<img src="https://github.com/mattignal/article-summary-details/blob/main/Abstract_details.png">
