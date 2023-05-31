import itertools
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import precision_score as precision_score_sklearn
from sklearn.metrics import recall_score as recall_score_sklearn
from sklearn.metrics import jaccard_score as jaccard_score_sklearn
from seqeval.metrics import f1_score,precision_score,recall_score


def f1_entity_level(*args, **kwargs):
    return f1_score(*args, **kwargs)


def f1_token_level(true_labels, predictions):
    true_labels = list(itertools.chain(*true_labels))
    predictions = list(itertools.chain(*predictions))
    
    labels = list(set(true_labels)) #- {'[PAD]', 'O'})
    
    return f1_score_sklearn(true_labels, 
                            predictions, 
                            average='macro',
                            labels=labels)
def precision_entity_level(*args, **kwargs):
    return precision_score(*args, **kwargs)


def precision_token_level(true_labels, predictions):
    true_labels = list(itertools.chain(*true_labels))
    predictions = list(itertools.chain(*predictions))
    
    labels = list(set(true_labels)) #- {'[PAD]', 'O'})
    
    return precision_score_sklearn(true_labels, 
                            predictions, 
                            average='macro',
                            labels=labels)

def recall_entity_level(*args, **kwargs):
    return recall_score(*args, **kwargs)


def recall_token_level(true_labels, predictions):
    true_labels = list(itertools.chain(*true_labels))
    predictions = list(itertools.chain(*predictions))
    
    labels = list(set(true_labels)) #- {'[PAD]', 'O'})
    
    return recall_score_sklearn(true_labels, 
                            predictions, 
                            average='macro',
                            labels=labels)

def Jaccard_score_token(true_labels,predictions):
  true_labels = list(itertools.chain(*true_labels))
  predictions = list(itertools.chain(*predictions))
  labels = list(set(true_labels)) #- {'[PAD]', 'O'})
  return jaccard_score_sklearn(true_labels, 
                            predictions, 
                            average='macro',
                            labels=labels)


