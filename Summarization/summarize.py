import argparse
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from os import truncate
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import json
from tqdm import tqdm

# nltk.download('punkt')
# nltk.download('stopwords')
if torch.cuda.is_available():
  torch_device="cuda:0"
else:
  torch_device="cpu"

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')#.to(torch_device)

def read_document(textfile):
  with open(textfile,"r") as f:
    text_corpus=f.read()
  text_corpus=text_corpus.replace("\n","")
  text_corpus=text_corpus.strip().split(".")
  
  print(text_corpus)
  return text_corpus
def preprocess(text):
  sentences=[]
  for t in text:
    sentences.extend(sent_tokenize(t))
  cleaned_sentences = pd.Series(sentences)#.str.replace("[^a-zA-Z]", " ", regex=True)
  #cleaned_sentences = [s.lower() for s in cleaned_sentences]
  
  #removing stop words
  #cleaned_sentences = [remove_stopwords(sent.split()) for sent in cleaned_sentences]
  print(cleaned_sentences)

  return cleaned_sentences 

def initialize_gloveembeddings(glove_location):
  glove_embeddings=dict()

  with open(glove_location,"r") as f:
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      glove_embeddings[word] = coefs
    #print(len(glove_embeddings.keys()))
  return glove_embeddings


def get_gloveembeddings(sentences,glove):
  sentence_vectors = []
  for i in sentences:
    if len(i) != 0:
      v = sum([glove.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)
  #print(sentence_vectors)
  return sentence_vectors

def generate_summary_textrank(sentences,sentence_vectors,sn):
  sim_matrix = np.zeros([len(sentences), len(sentences)])

  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i != j:
        sim_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

  nx_graph = nx.from_numpy_array(sim_matrix)

  scores = nx.pagerank(nx_graph)

  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
  summary = ""
  # Generate summary
  for i in range(sn):
    summary = summary + ranked_sentences[i][1] + ". "
  #print(summary)
  return summary

def bart_summarize(text):
    num_beams = 4
    #length_penalty = 2.0
    max_length = 800
    #min_length = 100 
    text = text.replace('\n','')
    text_input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024, truncation=True)['input_ids']#.to(torch_device)
    summary_ids = model.generate(text_input_ids, num_beams=int(num_beams), max_length=int(max_length))           
    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt
def save_True_statements(filename,textrank,bart,combined,folder_location=None):
  #print(folder_location,filename)
  summary1=textrank.split(". ")
  summary2=bart.split(". ")
  summary3=combined.split(". ")
  Summarized_text=[]
  Summarized_text.extend(summary1)
  Summarized_text.extend(summary2)
  Summarized_text.extend(summary3)
  Summarized_text=list(set(Summarized_text))
  #print(len(Summarized_text))
  save_folder="Summary"
  if folder_location:
    print(folder_location)
    save_path=os.path.join(folder_location,save_folder)
  else:
    save_path=os.path.join(os.getcwd(),save_folder)
  print(save_path)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    #print(save_path)
  filename=filename.replace(".txt","-Summarized.txt").split("/")[-1]
  save_path=os.path.join(save_path,filename)
  print(save_path)
  with open(save_path,"w") as f:
    for sent in Summarized_text:
      if len(sent.split())<=6:
        continue
      f.write(sent)
      f.write("\n")
    print("Saved Successfully at {}".format(save_path))

def extract_summary_dataset(Datafolder,glove_location):

  Methods=['Textrank','BART','combined']
  Metrics=['Rouge-1','Rouge-2','Rouge-L']
  Scores=list()
  glove=initialize_gloveembeddings(glove_location)
  for dir in tqdm(os.listdir(Datafolder)):
    Classfolder=os.path.join(Datafolder,dir)
    for dir2 in tqdm(os.listdir(Classfolder)):
      subfolder=os.path.join(Classfolder,dir2)
      #try:
      print(subfolder)
      Subject_roguescore=dict()
      Subject_roguescore["Class"]=dir
      Subject_roguescore["Subject"]=dir2
      Subject_roguescore["Method"]=dict()
      for method in Methods:
        Subject_roguescore["Method"][method]=dict()
        for metric in Metrics:
          Subject_roguescore["Method"][method][metric]=list()
      for file in os.listdir(os.path.join(subfolder,"afterCoref")):

        filename=file
        file_location=os.path.join(subfolder,"afterCoref",file)
        folder_location=subfolder
        print(file_location)
        corpus=read_document(file_location)
        #print(len(corpus))
        cleaned=preprocess(corpus)

        sentence_vectors=get_gloveembeddings(cleaned,glove)
        #Textrank Algorithm
        Num_sent_generate=min(len(cleaned),30)
        summary=generate_summary_textrank(cleaned,sentence_vectors,Num_sent_generate).strip(" ")
        test_corpus=". ".join(corpus)
        print("\n***TextRank Summary***\n")
        #print(summary)
        scoring = Rouge()


        print("\n***ROUGE Score***\n")
        Rogue_score=scoring.get_scores(summary, test_corpus)
        print(Rogue_score)
        Subject_roguescore["Method"]["Textrank"]["Rouge-1"].append(Rogue_score[0]['rouge-1']['f'])
        Subject_roguescore["Method"]["Textrank"]["Rouge-2"].append(Rogue_score[0]['rouge-2']['f'])
        Subject_roguescore["Method"]["Textrank"]["Rouge-L"].append(Rogue_score[0]['rouge-l']['f'])

        #Bart Summarization
        bs = bart_summarize(test_corpus)
        print("\n***BART Summary***\n")
        #print(bs)
        scoring = Rouge()
        print("\n***ROUGE Score***\n")
        Rogue_score=scoring.get_scores(bs, test_corpus)
        print(Rogue_score)
        Subject_roguescore["Method"]["BART"]["Rouge-1"].append(Rogue_score[0]['rouge-1']['f'])
        Subject_roguescore["Method"]["BART"]["Rouge-2"].append(Rogue_score[0]['rouge-2']['f'])
        Subject_roguescore["Method"]["BART"]["Rouge-L"].append(Rogue_score[0]['rouge-l']['f'])
        #print(scores)
        combined = bs + ". " +summary
        combinedsum = bart_summarize(combined)
        print("\n***Combined Summary***\n")
        #print(combinedsum)
        scoring = Rouge()
        print("\n***ROUGE Score***\n")
        Rogue_score=scoring.get_scores(combined, test_corpus)
        print(Rogue_score)
        Subject_roguescore["Method"]["combined"]["Rouge-1"].append(Rogue_score[0]['rouge-1']['f'])
        Subject_roguescore["Method"]["combined"]["Rouge-2"].append(Rogue_score[0]['rouge-2']['f'])
        Subject_roguescore["Method"]["combined"]["Rouge-L"].append(Rogue_score[0]['rouge-l']['f'])
        save_True_statements(filename,summary,bs,combinedsum,folder_location)
      #except:
      #  continue
      for method in Methods:
        for metric in Metrics:
          Subject_roguescore["Method"][method][metric]=sum(Subject_roguescore["Method"][method][metric])/len(Subject_roguescore["Method"][method][metric])

      Scores.append(Subject_roguescore)
    
    with open("./Summarizer_Metrics.json","w") as f:
      json.dump(Scores,f)


def extract_summary(file_location,glove_location,save_location):
  corpus=read_document(file_location)
  #print(len(corpus))

  cleaned=preprocess(corpus)
  glove=initialize_gloveembeddings(glove_location)
  sentence_vectors=get_gloveembeddings(cleaned,glove)
  #Textrank Algorithm
  Num_sent_generate=min(len(cleaned),30)
  test_corpus=". ".join(corpus)
  summary=generate_summary_textrank(cleaned,sentence_vectors,Num_sent_generate).strip(" ")
  print("\n***TextRank Summary***\n")
  print(summary)
  scoring = Rouge()
  print("\n***ROUGE Score***\n")
  Rogue_score=scoring.get_scores(summary, test_corpus)
  print(Rogue_score)
  #Bart Summarization
  bs = bart_summarize(test_corpus)
  print("\n***BART Summary***\n")
  #print(bs)
  scoring = Rouge()
  print("\n***ROUGE Score***\n")
  Rogue_score=scoring.get_scores(bs, test_corpus)
  print(Rogue_score)
  #print(scores)
  combined = bs + ". " +summary
  combinedsum = bart_summarize(combined)
  print("\n***Combined Summary***\n")
  #print(combinedsum)
  scoring = Rouge()
  print("\n***ROUGE Score***\n")
  Rogue_score=scoring.get_scores(combined, test_corpus)
  print(Rogue_score)
  
  save_True_statements(file_location,summary,bs,combinedsum,save_location)

parser = argparse.ArgumentParser()
parser.add_argument('--file_location', dest='file_location', default=None,
                    help='File location which to be extracted for summary')
parser.add_argument('--glove_location', dest='glove_location',default="/content/drive/MyDrive/FYP/Summarization/glove.6B.100d.txt",
                    help='File location of Glove Embedding File')
parser.add_argument('--folder_location', dest='folder_location', default=None,
                    help='Summarizing entire Folder content')
parser.add_argument('--save_location', dest='save_location', default=None,
                    help='Save Location for Summarized content')
args = parser.parse_args()
if args.file_location:
  extract_summary(args.file_location,args.glove_location,args.save_location)
elif args.folder_location:
  extract_summary_dataset(args.folder_location,args.glove_location)
else:
  print("Neither File nor Folder Location is given")