import torch
from pytorch_transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import copy
from tqdm import trange
import os
import logging
logger = logging.getLogger('sequence_tagger_bert')



class ModelTrainerBert:
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler,
                 train_dataset, 
                 val_dataset, 
                 update_scheduler='es', # ee(every_epoch) or every_step(es)
                 keep_best_model=False,
                 restore_bm_on_lr_change=False,
                 max_grad_norm=1.0,
                 smallest_lr=0.,
                 validation_metrics=None,
                 decision_metric=None,
                 loader_args={'num_workers' : 1},
                 batch_size=32,
                 model_type=None,
                 logger_dir=None):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
            
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        
        self._update_scheduler = update_scheduler
        self._keep_best_model = keep_best_model
        self._restore_bm_on_lr_change = restore_bm_on_lr_change
        self._max_grad_norm = max_grad_norm
        self._smallest_lr = smallest_lr
        self._validation_metrics = validation_metrics
        self._decision_metric = decision_metric
        if self._decision_metric is None:
            self._decision_metric = lambda metrics: metrics[0]
            
        self._loader_args = loader_args
        self._batch_size = batch_size
        self.logger_dir=os.path.join(logger_dir,model_type)
        self.writer = SummaryWriter(model_type)
    
    def _make_tensors(self, dataset_row):
        tokens, labels = tuple(zip(*dataset_row))
        return self._model.generate_tensors_for_training(tokens, labels)
    def update_dataset(self,dataset):
      self._train_dataset.extend(dataset)
    
    def train(self, epochs):
        #print(self._validation_metrics)
        best_model = {}
        best_dec_metric = float('inf')
        
        get_lr = lambda: self._optimizer.param_groups[0]['lr']
        
        train_dataloader = DataLoader(self._train_dataset, 
                                      batch_size=self._batch_size, 
                                      shuffle=True,
                                      collate_fn=self._make_tensors)
        iterator = trange(epochs, desc='Epoch')
        for epoch in iterator:
            self._model._bert_model.train()

            cum_loss = 0.
            for nb, tensors in enumerate(train_dataloader):
                loss = self._model.batch_loss_tensors(*tensors)
                cum_loss += loss.item()
                
                self._model._bert_model.zero_grad()
                loss.backward()
                if self._max_grad_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(parameters=self._model._bert_model.parameters(), 
                                                   max_norm=self._max_grad_norm)
                loss_running=cum_loss/(nb+1)    
                self._optimizer.step()
                self.writer.add_scalar("Loss/Running_train", loss_running, epoch * len(train_dataloader) + nb)
                if self._update_scheduler == 'es':
                    self._lr_scheduler.step()
            
            prev_lr = get_lr()
            logger.info(f'Current learning rate: {prev_lr}')
            print(f'Current learning rate: {prev_lr}')

            
            cum_loss /= (nb + 1)
            logger.info(f'Train loss: {cum_loss}')
            print(f'Train loss: {cum_loss}')
            self.writer.add_scalar("Loss/train", cum_loss, epoch)


            dec_metric = 0.
            if self._val_dataset is not None:
                _, __, val_metrics = self._model.predict(self._val_dataset, evaluate=True, 
                                                         metrics=self._validation_metrics)
                val_loss = val_metrics[0]
                logger.info(f'Validation loss: {val_loss}')
                print(f'Validation loss: {val_loss}')
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                logger.info(f'Validation metrics: {val_metrics[1:]}')
                #print(f'Validation metrics: F1_score-entity - {val_metrics[1:][0]}')
                print(f'Validation metrics: F1_score-token - {val_metrics[1:][1]}')
                #print(f'Validation metrics: Recall-entity - {val_metrics[1:][2]}')
                print(f'Validation metrics: Recall-token - {val_metrics[1:][3]}')
                #print(f'Validation metrics: Precision-entity - {val_metrics[1:][4]}')
                print(f'Validation metrics: Precision-token - {val_metrics[1:][5]}')
                print(f'Validation metrics: Jaccard Score - {val_metrics[1:][6]}')
                #self.writer.add_scalar("Metrics/F1_entity",val_metrics[1:][0],epoch)
                self.writer.add_scalar("Metrics/F1_token",val_metrics[1:][1],epoch)
                #self.writer.add_scalar("Metrics/Recall_entity",val_metrics[1:][2],epoch)
                self.writer.add_scalar("Metrics/Recall_token",val_metrics[1:][3],epoch)
                #self.writer.add_scalar("Metrics/Precision_entity",val_metrics[1:][4],epoch)
                self.writer.add_scalar("Metrics/Precision_token",val_metrics[1:][5],epoch)
                self.writer.add_scalar("Metrics/Jaccard Index",val_metrics[1:][6],epoch)
                dec_metric = self._decision_metric(val_metrics)
                
                if self._keep_best_model and (dec_metric < best_dec_metric):
                    best_model = copy.deepcopy(self._model._bert_model.state_dict())
                    best_dec_metric = dec_metric
            
            if self._update_scheduler == 'ee':
                self._lr_scheduler.step(dec_metric)
                
            if self._restore_bm_on_lr_change and get_lr() < prev_lr:
                if get_lr() < self._smallest_lr: 
                    iterator.close()
                    break

                prev_lr = get_lr()
                logger.info(f'Reduced learning rate to: {prev_lr}')
                print(f'Reduced learning rate to: {prev_lr}')
                    
                logger.info('Restoring best model...')
                print('Restoring best model...')
                self._model._bert_model.load_state_dict(best_model)

        if best_model:
            self._model._bert_model.load_state_dict(best_model)

        torch.cuda.empty_cache()
