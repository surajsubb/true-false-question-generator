from sequence_tagger_bert import SequenceTaggerBert
from bert_for_token_classification_custom import *
from model_trainer_bert import ModelTrainerBert
from transformers import BertTokenizer,AlbertTokenizer,AutoTokenizer,DistilBertTokenizer,ConvBertTokenizer,MobileBertTokenizer

from bert_utils import get_model_parameters
from bert_utils import *
from model_trainer_bert import ModelTrainerBert
from metrics import *
import os
from pytorch_transformers import AdamW, WarmupLinearSchedule
import argparse
import pandas as pd
import logging
import random
import sys
import json
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('sequence_tagger_bert')

parser = argparse.ArgumentParser()
parser.add_argument('--Train_dataset', dest='train_dataset', default="/content/gdrive/MyDrive/FYP/Answer Identification/tqa_a_train.csv",
                    help='File location of the Training dataset')
parser.add_argument('--Dev_dataset', dest='val_dataset',default="/content/gdrive/MyDrive/FYP/Answer Identification/tqa_a_dev.csv",
                    help='File location of Validation set')
parser.add_argument('--Test_dataset', dest='test_dataset', default="/content/gdrive/MyDrive/FYP/Answer Identification/tqa_a_test.csv",
                    help='File location of test_dataset')
parser.add_argument('--Unlabeled_dataset', dest='Unlabeled_dataset', default="/content/gdrive/MyDrive/FYP/Dataset",
                    help='File location of created text corpus')
parser.add_argument('--batch_size', dest='batch_size',default=8,
                    help='batch size for training')

parser.add_argument('--epochs', dest='epochs', default=5,type=int,
                    help='Number of Epochs of Training')
parser.add_argument('--model_type', dest='model_type', default='bert',
                    help='model type to be used in training')
parser.add_argument('--model_name', dest='model_name', default='bert-base-cased',
                    help='model to be used in training')
parser.add_argument('--mode', dest='mode',default="train",
                    help='Mode of Training or Testing')
parser.add_argument('--lr', dest='lr',default=5e-5,
                    help='Learning Rate for training')

parser.add_argument('--N_splits', dest='N_splits', default=2,
                    help='Splits of Unlabeled Dataset for Semisupervised training')
parser.add_argument('--logger', dest='log_dir', default="./log_dir",
                    help='Logger directory')
parser.add_argument('--save_location', dest='save_location', default="./saved_model",
                    help='Directory for saving and loading trained modela')
args = parser.parse_args()


def update_tag(tag):
  if tag=="B" or tag=="I":
    tag="A"
  return tag

def make_corpus(dataset_path):
  train_data=pd.read_csv(dataset_path)
  train_data=train_data[["Text_tok","Gold_iob"]]
  train_corpus=zip(train_data["Text_tok"],train_data["Gold_iob"])
  return train_corpus

def Self_Trainer_Sampler(unlabeled_dataset,splits):
  Unlabeled=[]
  Dataset_folder=unlabeled_dataset
  Classes=["Class9","Class8"] #to be added
  Subjects=["Geography","Biology"]  # to be added
  for Class in Classes:
    for Sub in Subjects:
      path=os.path.join(Dataset_folder,Class,Sub,"Summary")
      for files in os.listdir(path):
        file_path=os.path.join(path,files)
        with open(file_path,"r") as f:
          for line in f.readlines():
            Unlabeled.append(line.strip().split())
  N_splits=splits
  Sampler=Unlabeled
  #Sampler=[]
  #import numpy as np
  #from sklearn.model_selection import KFold
  #kf = KFold(n_splits=N_splits,random_state=100,shuffle=True)
  #kf.get_n_splits(Unlabeled)
  #for i, (train_index, test_index) in enumerate(kf.split(Unlabeled)):
  #  Sampler.append(list(map(Unlabeled.__getitem__, test_index)))
  return Sampler


def update_dataset(dataset):
  updated_dataset=[]
  for i,(text,tags) in enumerate(dataset):
    updated_tags=[update_tag(t) for t in tags]
    updated_dataset.append((text,updated_tags))
  return updated_dataset


def make_predictions_answers(Dataset_folder,Classes,Subjects,model):
  Answer_predictions=list()
  Answer_nopredictions=list()
  for Class in Classes:
    for Sub in Subjects:
      path=os.path.join(Dataset_folder,Class,Sub,"Summary")
      for files in os.listdir(path):
        file_path=os.path.join(path,files)
        chapter=file_path.split("_")
        with open(file_path,"r") as f:
          for line in f.readlines():
            data={}
            sent=line.split()

            pred=model.predict([sent])[0][0]
            data["Subject"]=Sub
            data["Class"]=Class
            data["Chapter"]=chapter[2]
            data["Statement"]=line
            data["Tokens"]=sent
            data["Predictions"]=pred
            if 'A' in pred:
              data["Mask_index"]=[ind for ind,val in enumerate(pred) if val=='A']
              data["Mask_Token"]=[sent[i] for i in data["Mask_index"]]
              Answer_predictions.append(data)
            else:
              Answer_nopredictions.append(data)

            print(sent)
            print(pred)

  with open("./jsons/Answerpredictions.json","w") as f1:
    json.dump(Answer_predictions,f1)
  with open("./jsons/NoAnswerpredictions.json","w") as f2:
    json.dump(Answer_nopredictions,f2)


def make_predictions_answers_file(file_path,model):
  Answer_predictions=list()
  Answer_nopredictions=list()
  with open(file_path,"r") as f:
    for line in f.readlines():
      data={}
      sent=line.split()
      pred=model.predict([sent])[0][0]
      #data["Subject"]=Sub
      #data["Class"]=Class
      #data["Chapter"]=chapter[2]
      data["Statement"]=line
      data["Tokens"]=sent
      data["Predictions"]=pred
      if 'A' in pred:
        data["Mask_index"]=[ind for ind,val in enumerate(pred) if val=='A']
        data["Mask_Token"]=[sent[i] for i in data["Mask_index"]]
        Answer_predictions.append(data)
      else:
        Answer_nopredictions.append(data)
      print(sent)
      print(pred)

  with open("./jsons/Answerpredictions.json","w") as f1:
    json.dump(Answer_predictions,f1)
  with open("./jsons/NoAnswerpredictions.json","w") as f2:
    json.dump(Answer_nopredictions,f2)

def main(args):

  model_type = args.model_type
  model_name=args.model_name
  #bpe_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
  VOCAB=('[PAD]','O','A')
  tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
  idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

  if args.mode=='train':
    pseudo=[]
    train_corpus=make_corpus(args.train_dataset)
    validation_corpus=make_corpus(args.val_dataset)
    test_corpus=make_corpus(args.test_dataset)
    batch_size = args.batch_size
    n_epochs = args.epochs
    train_dataset = prepare_tqaa_corpus(train_corpus)
    val_dataset = prepare_tqaa_corpus(validation_corpus)
    test_dataset=prepare_tqaa_corpus(test_corpus)
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)
    Sampler=Self_Trainer_Sampler(args.Unlabeled_dataset,args.N_splits)
    print(train_dataset[0:10])
    updated_train_dataset=update_dataset(train_dataset)
    updated_val_dataset=update_dataset(val_dataset)
    updated_test_dataset=update_dataset(test_dataset)
    print(updated_train_dataset[0:10])
    for i in range(args.N_splits):
      if model_type=="bert":
        model = BertForTokenClassificationCustom.from_pretrained(model_name, 
                                                             num_labels=len(tag2idx)).cuda()
        bpe_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
      elif model_type=="albert":
        model = AlbertForTokenClassificationCustom.from_pretrained(model_name, 
                                                             num_labels=len(tag2idx)).cuda()
        bpe_tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
      elif model_type=="convbert":
        model = ConvbertForTokenClassificationCustom.from_pretrained(model_name, 
                                                               num_labels=len(tag2idx)).cuda()
        bpe_tokenizer = ConvBertTokenizer.from_pretrained(model_name, do_lower_case=False)
      elif model_type=="distilbert":
        model = DistilbertForTokenClassificationCustom.from_pretrained(model_name, 
                                                               num_labels=len(tag2idx)).cuda()
        bpe_tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=False)
      else:
        model = MobilebertForTokenClassificationCustom.from_pretrained(model_name, 
                                                               num_labels=len(tag2idx)).cuda()
        bpe_tokenizer = MobileBertTokenizer.from_pretrained(model_name, do_lower_case=False)
      
      seq_tagger = SequenceTaggerBert(bert_model=model, bpe_tokenizer=bpe_tokenizer, 
                                idx2tag=idx2tag, tag2idx=tag2idx, max_len=256,
                                pred_batch_size=batch_size)
      optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
      #optimizer = AdamW(get_model_parameters(model,no_decay={'bias', 'gamma', 'beta'},full_finetuning=True), lr=5e-4, betas=(0.9, 0.999), 
      #                  eps=1e-6, weight_decay=0.01, correct_bias=True)

      n_iterations_per_epoch = len(updated_train_dataset) / batch_size
      n_steps = n_iterations_per_epoch * n_epochs
      print(n_steps)
      lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=250, t_total=n_steps)
      print(len(updated_train_dataset))
      trainer = ModelTrainerBert(model=seq_tagger, 
                                 optimizer=optimizer, 
                                 lr_scheduler=lr_scheduler,
                                 train_dataset=updated_train_dataset, 
                                 val_dataset=updated_val_dataset,
                                 validation_metrics=[f1_entity_level,f1_token_level,recall_entity_level,recall_token_level,precision_entity_level,precision_token_level,Jaccard_score_token],
                                 batch_size=batch_size,
                                 model_type=model_type,
                                 logger_dir=args.log_dir)


      trainer.train(epochs=n_epochs)
      print(Sampler[0:10],len(Sampler))
      for sent in Sampler:
          pseudo.append((sent,seq_tagger.predict([sent])[0][0]))
      updated_train_dataset.extend(pseudo)
      print(len(updated_train_dataset))
    _, __, test_metrics = seq_tagger.predict(updated_val_dataset, evaluate=True, 
                                         metrics=[f1_entity_level,f1_token_level,recall_entity_level,recall_token_level,precision_entity_level,precision_token_level,Jaccard_score_token])

    print(f'Token-level f1: {test_metrics[2]}')
    print(f'Token-level recall: {test_metrics[4]}')
    print(f'Token-level precision: {test_metrics[6]}')
    print(f'Jaccard Score: {test_metrics[7]}')

    _, __, test_metrics = seq_tagger.predict(updated_test_dataset, evaluate=True, 
                                         metrics=[f1_entity_level,f1_token_level,recall_entity_level,recall_token_level,precision_entity_level,precision_token_level,Jaccard_score_token])

    print(f'Token-level f1: {test_metrics[2]}')
    print(f'Token-level recall: {test_metrics[4]}')
    print(f'Token-level precision: {test_metrics[6]}')
    print(f'Jaccard Score: {test_metrics[7]}')

    save_path=os.path.join(args.save_location,args.model_type)
    seq_tagger.save_serialize(save_path)
    
    
    if model_type=="bert":
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,BertForTokenClassificationCustom)

    elif model_type=="albert":
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,AlbertForTokenClassificationCustom)
    elif model_type=="convbert":
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,ConvbertForTokenClassificationCustom)
    elif model_type=="distilbert":
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,DistilbertForTokenClassificationCustom)

    else:
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,MobilebertForTokenClassificationCustom)
    #make_predictions_answers(args.Unlabeled_dataset,["Class9","Class8"],["Geography","Biology"],seq_tagger1)
  else:
    save_path=os.path.join(args.save_location,args.model_type)

    if model_type=="bert":
      bpe_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,BertForTokenClassificationCustom)

    elif model_type=="albert":
      bpe_tokenizer = AlbertTokenizer.from_pretrained("/media/suraj/New Volume/suraj/college/8th sem/FYP/Final Application/FYP-20230413T131216Z-001/FYP/Answer Identification", do_lower_case=False)
      print(bpe_tokenizer)
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,AlbertForTokenClassificationCustom)
    elif model_type=="convbert":
      bpe_tokenizer = ConvBertTokenizer.from_pretrained(model_name, do_lower_case=False)
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,ConvbertForTokenClassificationCustom)
    elif model_type=="distilbert":
      bpe_tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=False)
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,DistilbertForTokenClassificationCustom)

    else:
      bpe_tokenizer = MobileBertTokenizer.from_pretrained(model_name, do_lower_case=False)
      seq_tagger1 = SequenceTaggerBert.load_serialized(save_path,MobilebertForTokenClassificationCustom)
    print("loaded successfully")
    make_predictions_answers_file(args.Unlabeled_dataset,seq_tagger1)

main(args)


