# Import Statements
import math
import numpy as np
import re
from newspaper import Article
from textwrap import TextWrapper
from spacy.lang.en import English
import torch
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, BartConfig
import nltk
nltk.download('punkt')

wrapper = TextWrapper(width=80)
# cnn_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# cnn_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

cnn_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")

cnn_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")

def get_article(url):
  """Get info about article"""
  article = Article(url)
  article.download()
  article.parse()
  text = article.text
  title = article.title
  authors = article.authors
  if authors == []:
      authors = ["Unknown"]
  try:
      date = str(article.publish_date.date())
  except AttributeError:
      date = "Unknown"
  return article, text, title, authors, date
  
def create_paragraphs(text):
  """Buckets into paragraphs for analysis"""
  paragraphs = text.split('\n\n')
  paragraphs = [x for x in paragraphs if len(x) > 100] # must be > 100 characters (assume else is heading or irrelevant)
  return paragraphs

def drop_paragraphs(paragraphs, list_to_drop):
  """function to allow the user to remove paragraphs they feel are unimportant"""
  for i in sorted(list_to_drop, reverse=True):
    del paragraphs[i - 1]
  return paragraphs
 
def create_abstract(paragraphs, title, authors, date):
  article_cleaned = " ".join(paragraphs)
  inputs = cnn_tokenizer([article_cleaned], max_length=1024, truncation=True, # limited to first 1024 tokens
                         return_tensors='pt')
  summary_ids = cnn_model.generate(inputs['input_ids'], num_return_sequences=1,
                                  early_stopping=True, num_beams=3,
                                  min_length=80, max_length=120, 
                                  do_sample=False)
  abstract = cnn_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
  abstract = re.sub(r", writes ([^\s]+) ([^\s]+).", ", writes {}.".format(authors[0]), abstract) # sometimes misses authors name
  abstract = re.sub(r", says ([^\s]+) ([^\s]+)", "", abstract) # sometimes misses authors name
  return abstract
  
def chunk_paragraphs(paragraphs, granularity=2):
  """Chunks paragraphs into start, end, and then a series of middle paragraphs
  param granularity: controls level of detail, more granularity may mean more paragraphs to process
  """
  if len(paragraphs) >= 6:
    block_off = 2
    middle = paragraphs[block_off:-block_off]
  elif len(paragraphs) >= 4:
    block_off = 1
    middle = paragraphs[block_off:-block_off]
  else:
    block_off = 0
    middle = paragraphs

  lengths = []
  chunks = []
  paragraphs_to_chunk = []
  present_length = 0
  for paragraph in middle:
    inputs = cnn_tokenizer([paragraph], return_tensors='pt', truncation=True)
    length = len(inputs['input_ids'][0])
    lengths.append(length)
    avg_length = np.mean(lengths)
    present_length += length
    if present_length > 1024:
      chunks.append(paragraphs_to_chunk)
      present_length = 0
    elif present_length >= 1024 - avg_length*granularity:
      paragraphs_to_chunk.append(paragraph)
      chunks.append(paragraphs_to_chunk)
      paragraphs_to_chunk = []
      present_length = 0
    else:
        paragraphs_to_chunk.append(paragraph)

  if len(chunks) == 0:
    chunks = [paragraphs_to_chunk]

  if block_off != 0: 

      start_chunks = " ".join(paragraphs[:block_off])

      last_chunk = " ".join(chunks[-1])
      end_chunks = " ".join(paragraphs[-block_off:])
      inputs = cnn_tokenizer([last_chunk], return_tensors='pt', truncation=True)
      lc_length = len(inputs['input_ids'][0])
      inputs = cnn_tokenizer([end_chunks], return_tensors='pt', truncation=True)
      ec_length = len(inputs['input_ids'][0])
      if lc_length + ec_length <= 1024 and ec_length <= 1024 - avg_length*granularity:
        # print("Adding final 'middle chunk' to the end of article chunk.")
        end_chunks = last_chunk + " " + end_chunks
        chunks = chunks[:-1]

      if len(chunks) > 0:
        chunks = [". ".join(x) for x in chunks]
        paragraph_chunks = [start_chunks] + chunks + [end_chunks]

      else:
          paragraph_chunks = [start_chunks] + [end_chunks]
  else:
      paragraph_chunks = [". ".join(x) for x in chunks]

  return paragraph_chunks
  
def key_details(paragraph_chunks):
  """Gets key ideas and details from each part of article"""
  details_list = []
  for chunk in paragraph_chunks:
    key_idea, details = get_key_details(chunk)
    details_list.append(details)
  return details_list

def get_key_details(chunk):
    inputs = cnn_tokenizer([chunk], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = cnn_model.generate(inputs['input_ids'], num_return_sequences=1, output_scores=False, 
                                  early_stopping=True, num_beams=3, length_penalty=0.2)
    key_idea = [cnn_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
    key_idea = re.sub(r'(\"[^\"]*)\" [A-Z]', r'\1." ', key_idea[0]) # second quotation(") fix
    key_idea = key_idea.replace('. “', '.“') # additional quotation fix
    key_idea = re.sub(r", writes ([^\s]+) ([^\s]+).", ".", key_idea) # sometimes misses authors name
    key_idea = re.sub(r", says ([^\s]+) ([^\s]+).", ".", key_idea) # sometimes misses authors name
    key_idea = key_idea.replace(", he says", "") # sometimes misses authors name
    key_idea = re.sub(r"\. \b[A-Z].*?\b\: ", ". ", key_idea) # sometimes misses authors name
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    key_idea = [sent.string.strip() for sent in nlp(key_idea).sents][0]
    summary_ids = cnn_model.generate(inputs['input_ids'], num_return_sequences=1,
                                  early_stopping=True, num_beams=2, 
                                  min_length=80, 
                                  max_length=160, 
                                  do_sample=False)
    details = [cnn_tokenizer.decode(g, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True) for g in summary_ids][0]

    # if the key idea is present in the details, let's first look for an alternative generation
    if key_idea in details:
      summary_ids = cnn_model.generate(inputs['input_ids'], num_return_sequences=1,
                                  early_stopping=True, num_beams=2, 
                                  min_length=80, 
                                  max_length=160, 
                                  top_p = 0.8,
                                  do_sample=True)
      details_alt = [cnn_tokenizer.decode(g, skip_special_tokens=True, 
                                           clean_up_tokenization_spaces=True) for g in summary_ids]
      
      # if the key idea is present in the alternative, just use the original and remove the key idea
      if key_idea in details_alt[0]:
        details = details.replace(key_idea, "")
      else:
        details = details_alt[0]
    
    details = re.sub(r", writes ([^\s]+) ([^\s]+).", ".", details) # sometimes misses authors name
    details = re.sub(r", says ([^\s]+) ([^\s]+).", ".", details) # sometimes misses authors name
    details = details.replace(", he says", "") # sometimes misses authors name
    details = re.sub(r"\. \b[A-Z].*?\b\: ", ". ", details) # sometimes misses authors name
    return key_idea, details
    
def generate_summary(details_list, authors, granularity=2):
  """generates summary"""
  paragraph_chunks = chunk_paragraphs(details_list, granularity=granularity)
  summaries = []
  for chunk in paragraph_chunks:
    summary = get_summary(chunk)
    summaries.append(summary)
  return summaries

def get_summary(chunk, authors):
    inputs = cnn_tokenizer([chunk], max_length=1024, truncation=True, # limited to first 1024 tokens
                          return_tensors='pt')
    summary_ids = cnn_model.generate(inputs['input_ids'], num_return_sequences=1,
                                    early_stopping=True, num_beams=3,
                                    min_length=80, max_length=120, 
                                    do_sample=False)
    summary = cnn_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    summary = re.sub(r", writes ([^\s]+) ([^\s]+).", ", writes {}.".format(authors[0]), summary) # sometimes misses authors name
    summary = re.sub(r", says ([^\s]+) ([^\s]+)", "", summary)
    summary = re.sub(r"\. \b[A-Z].*?\b\: ", ". ", summary) # sometimes misses authors name
    return summary
