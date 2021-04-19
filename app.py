import streamlit as st
import os
# import torch
import nltk
import urllib.request
from newspaper import Article
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, BartConfig
from apply_bart import get_article, create_abstract, drop_paragraphs, create_paragraphs, chunk_paragraphs, get_key_details, get_summary


def main():
    st.markdown("<h1 style='text-align: center;'>BART Abstractive Summarization</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Automatically generates Abstract, Key Ideas, Details, and Summary using BART for language comprehension</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Add Article</h3>", unsafe_allow_html=True)

    article_type = st.radio("Add Article: (minimum 2 paragraphs, recommend more) ", ["Text", "Enter URL"])

    if article_type == "Text":
        text = st.text_area("", "The Bart model was proposed in BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.According to the abstract, Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.\n\nFrom the abstract: We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and many other more recent pretraining schemes. We evaluate a number of noising approaches, finding the best performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine translation, with only target language pretraining. We also report ablation experiments that replicate other pretraining schemes within the BART framework, to better measure which factors most influence end-task performance.", 400)
        title = ""
        authors = [""]
        date = ""
    else:
        url = st.text_input("", "https://monthlyreview.org/2009/04/01/the-credit-crisis-is-the-international-role-of-the-dollar-at-stake/")
        article, text, title, authors, date = get_article(url)
        st.markdown(f"**Title**: {title}")
        st.markdown(f"**Authors**: {str(authors)}")
        st.markdown(f"**Date Published**: {date}")
        st.markdown(f"[**Article Link**]({url})")
    paragraphs = create_paragraphs(text)
    
    # Remove Paragraphs
    st.markdown("<h4 style='text-align: center;'>Recommended: Remove Irrelevant Paragraphs</h4>", unsafe_allow_html=True)
    my_expander1 = st.beta_expander(f"Show Paragraphs - Recall which #s to drop")
    with my_expander1:
        for index, paragraph in enumerate(paragraphs):
            st.text((index+1, paragraph[:100]))
    list_to_drop = st.text_input('List of paragraphs to drop (e.g. 1, 3, 5, 6)', '')
    if list_to_drop != "":
        list_to_drop2 = [int(x) for x in list_to_drop.split(",")]
        paragraphs = drop_paragraphs(paragraphs, list_to_drop2)

    st.markdown("<h3 style='text-align: center;'>Abstract</h3>", unsafe_allow_html=True)
    abstract = create_abstract(paragraphs, title, authors, date)
    st.markdown(f"<p align='justify'>{abstract}</p>", unsafe_allow_html=True)

        
    # Chunk paragraphs
    st.markdown("<h4 style='text-align: center;'>Chunk Paragraphs</h4>", unsafe_allow_html=True)
    granularity = st.slider("Set granularity", 1, 20, 5)
    paragraph_chunks = chunk_paragraphs(paragraphs, granularity=granularity)
    

    # Get Key Details
    st.markdown("<h3 style='text-align: center;'>Key Details</h3>", unsafe_allow_html=True)
    details_list = []
    for chunk in paragraph_chunks:
        key_idea, details = get_key_details(chunk)
        my_expander2 = st.beta_expander(f"{key_idea}")
        with my_expander2:
    	    st.markdown(f"{details}")
        details_list.append(details)
    
    # Generate Summary
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    paragraph_chunks = chunk_paragraphs(details_list, granularity=granularity)
    for chunk in paragraph_chunks:
        summary = get_summary(chunk, authors)
        st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)


#     # Summarize
#     sum_level = st.radio("Output Length: ", ["Short", "Medium"])
#     max_length = 3 if sum_level == "Short" else 5
#     result_fp = 'results/summary.txt'
#     summary = summarize(input_fp, result_fp, model, max_length=max_length)
#     st.markdown("<h3 style='text-align: center;'>Summary</h3>")
#     st.markdown(f"<p align='justify'>{summary}</p>")


# def download_model():
#     nltk.download('popular')
#     url = 'https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading checkpoint...")
#         progress_bar = st.progress(0)
#         with open('checkpoints/mobilebert_ext.pt', 'wb') as output_file:
#             with urllib.request.urlopen(url) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
#                         (counter / MEGABYTES, length / MEGABYTES))
#                     progress_bar.progress(min(counter / length, 1.0))

#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()


# @st.cache(suppress_st_warning=True)
# def load_model(model_type):
#     checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
#     model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=model_type)
#     return model





if __name__ == "__main__":
    main()
