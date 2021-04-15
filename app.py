import streamlit as st
import os
import torch
import nltk
import urllib.request
from newspaper import Article
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, BartConfig
from apply_bart import get_article, drop_paragraphs, chunk_paragraphs, key_details, generate_summary


def main():
    st.markdown("<h1 style='text-align: center;'>BART Abstractive Summarization✏️</h1>", unsafe_allow_html=True)

    # initialize BART-CNN
    cnn_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    cnn_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # Input
    input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
    st.markdown("<h3 style='text-align: center;'>Input</h3>", unsafe_allow_html=True)

    if input_type == "Raw Text":
        with open("raw_data/input.txt") as f:
            sample_text = f.read()
        article = st.text_area("", sample_text, 400)
        title = ""
        authors = []
        date = ""
    else:
        url = st.text_input("", "https://monthlyreview.org/2009/04/01/the-credit-crisis-is-the-international-role-of-the-dollar-at-stake/")
        st.markdown(f"[*View Article*]({url})")
        article, title, authors, date = get_article(url)
    
    st.markdown("<h1 style='text-align: center;'>Abstract✏️</h1>", unsafe_allow_html=True)
    create_abstract(paragraphs, title, authors, date)
    
    # Remove Paragraphs
    st.markdown("<h3 style='text-align: center;'>Optional: Remove Paragraphs</h3>", unsafe_allow_html=True)
    list_to_drop = st.text_input('List to drop', 'e.g. 1, 2, 6')
    paragraphs = drop_paragraphs(paragraphs, list(list_to_drop))
        
    # Create paragraphs
    st.markdown("<h3 style='text-align: center;'>Create Paragraphs</h3>", unsafe_allow_html=True)
    granularity = st.slider("Granularity", 1, 100)
    paragraph_chunks = chunk_paragraphs(paragraphs, granularity=2)
    
    # Get Key Details
    st.markdown("<h3 style='text-align: center;'>Create Paragraphs</h3>", unsafe_allow_html=True)
    details_list = key_details(paragraph_chunks)
    
    # Generate Summary
    generate_summary(details_list, authors, granularity=2)
    

#     # Summarize
#     sum_level = st.radio("Output Length: ", ["Short", "Medium"])
#     max_length = 3 if sum_level == "Short" else 5
#     result_fp = 'results/summary.txt'
#     summary = summarize(input_fp, result_fp, model, max_length=max_length)
#     st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
#     st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)


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
