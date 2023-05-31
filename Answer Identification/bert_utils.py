from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

def preprocess(t):
  t1=t.strip('][').split(', ')
  t_cleaned=[t.replace("'","")for t in t1]
  return t_cleaned

def prepare_tqaa_corpus(corpus, name='ner', filter_tokens={'-DOCSTART-'}):
    result = []
    for token,label in corpus:
        token=preprocess(token)
        label=preprocess(label)
        #token=row["Text"].split()
        #label=preprocess(row["Gold_iob"])
        result.append((token,label))
    return result

def prepare_conll_corpus(corpus,name='ner',filter_tokens={'-DOCSTART-'}):
    result=[]
    T=[]
    L=[]
    with open(corpus,"r") as f:
        t=[]
        l=[]
        for line in f:
            if len(line)>1:
                tokens=line.split()      
                token=tokens[0]
                label=tokens[-1]
                if token in filter_tokens:
                  continue
                else:
                  t.append(token)
                  l.append(label)
            else:
              T.append(t)
              L.append(l)
              t=[]
              l=[]

    for t,l in zip(T,L):
        if len(t)>0:
            result.append((t,l))
    return result


def get_parameters_without_decay(model, no_decay={'bias', 'gamma', 'beta'}):
    params_no_decay = []
    params_decay = []
    for n, p in model.named_parameters():
        if any((e in n) for e in no_decay):
            params_no_decay.append(p)
        else:
            params_decay.append(p)
    
    return [{'params' : params_no_decay, 'weight_decay' : 0.},
            {'params' : params_decay}]


def get_model_parameters(model, no_decay={'bias', 'gamma', 'beta'}, 
                         full_finetuning=True, lr_head=None):
    grouped_parameters = get_parameters_without_decay(model.classifier, no_decay)
    if lr_head is not None:
        for param in grouped_parameters:
            param['lr'] = lr_head
    
    if full_finetuning:
        grouped_parameters = (get_parameters_without_decay(model.bert, no_decay) 
                              + grouped_parameters)
    
    return grouped_parameters



    