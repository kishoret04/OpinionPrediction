import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import pytorch_lightning as pl
from transformers import DistilBertModel, BertTokenizerFast
from transformers import  ElectraModel
import transformers
from datetime import datetime

class DynamicFPModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = None
        self.config = config
        self.save_hyperparameters(config)
        self.train_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0], 'emotion': [0], 'author': [0], 'mean': [0]}
        self.dev_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0], 'emotion': [0], 'author': [0], 'mean': [0]}
        self.test_loss = {'vader': [0], 'flair': [0], 'sent': [0], 'subj': [0], 'emotion': [0], 'author': [0], 'mean': [0]}
       
        self.train_perf = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0, 'emotion': 0, 'mean': 0}
        self.dev_perf   = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,'emotion': 0, 'mean': 0}
        self.test_perf  = {'vader': 0, 'flair': 0, 'sent': 0, 'subj': 0,'emotion': 0, 'mean': 0}
        self.naspect = 0
        ###############~_update#######################################
        ##BERT pretrained
        # import BERT-base pretrained model
        # Feed input to BERT/ELECTRA
        if self.config['sel_model'] == 'bert':
            self.track_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            self.track_embedder = ElectraModel.from_pretrained('google/electra-small-discriminator')
        
        # Freeze the BERT/ELECTRA model
        if config['freeze_bert']:
            for param in self.track_embedder.parameters():
                param.requires_grad = False

        #####################################################################
        author_final_dim = 0

        if config['build_author_track']:
            input_size = config['hid_dim']
            if config['build_sentiment_embedding']:
                input_size += 4 * config['sentiment_dim']
            if config['leverage_emotion']:
                input_size += 6

            self.timestamp_merge = nn.Sequential(
                nn.Linear(input_size, config['author_track_dim']),
                nn.ReLU(),
                nn.Linear(config['author_track_dim'], config['author_track_dim']),
                nn.ReLU())

            self.track_encoder_read = getattr(nn, config['rnn_type'].upper())(
                input_size=config['author_track_dim'], hidden_size=config['author_track_dim'],
                num_layers=config['rnn_layer_read'], dropout=config['dropout'],
                batch_first=True, bidirectional=False)
            self.track_encoder_write = getattr(nn, config['rnn_type'].upper())(
                input_size=config['author_track_dim'], hidden_size=config['author_track_dim'],
                num_layers=config['rnn_layer_write'], dropout=config['dropout'],
                batch_first=True, bidirectional=False)

            #since read and write track are spearately encoded and merged
            author_final_dim += config['author_track_dim'] * config['rnn_layer_read']
            author_final_dim += config['author_track_dim'] * config['rnn_layer_write']
        
        in_dim = author_final_dim + config['hid_dim'] # target article
        self.dropout = nn.Dropout(config['dropout'])

        if config['sentiment_fingerprinting']:
            self.vader_predict = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.flair_predict = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.blob_sent = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.blob_subj = nn.Sequential(
                nn.Linear(in_dim, config['sentiment_dim']),
                nn.ReLU(),
                nn.Linear(config['sentiment_dim'], 3))
            self.naspect += 4

        if config['emotion_fingerprinting']:
            self.emotion_predict = nn.Linear(in_dim, config['emotion_dim'])
            self.naspect += 1

        if config['build_sentiment_embedding']:
            self.vader_embed = nn.Embedding(3, config['sentiment_dim'])
            self.vader_embed.weight = self.vader_predict[2].weight
            self.flair_embed = nn.Embedding(3, config['sentiment_dim'])
            self.flair_embed.weight = self.flair_predict[2].weight
            self.sent_embed = nn.Embedding(3, config['sentiment_dim'])
            self.sent_embed.weight = self.blob_sent[2].weight
            self.subj_embed = nn.Embedding(3, config['sentiment_dim'])
            self.subj_embed.weight = self.blob_subj[2].weight

        self.train_accuracy = torch.nn.ModuleList( [AspectACC(aspect=i) for i in range( self.naspect)] )
        self.val_accuracy = torch.nn.ModuleList( [AspectACC(aspect=i) for i in range( self.naspect)] )
        self.test_accuracy = torch.nn.ModuleList( [AspectACC(aspect=i) for i in range( self.naspect)] )
        # f1 score metrics
        self.val_f1 = torch.nn.ModuleList( [pl.metrics.F1(num_classes = 3 ) for i in range( self.naspect)] )
        self.test_f1 = torch.nn.ModuleList( [ pl.metrics.F1(num_classes = 3 ) for i in range( self.naspect)] )
        
    def forward(self,  author, read_track, write_track, article_pack, sentiments, emotion):
        result = {}        
        len_track = self.config['previous_comment_cnt']+1
        batch_size = read_track.shape[1]#author.size()[0]
        seq_len = torch.tensor( [len_track]* batch_size, device=self.device)
        self.seq_len = seq_len
        r_ht = self._bert_encoder(read_track[0], read_track[1])
        #Extracting last article to predict the sentiment
        final_idx = (seq_len - 1).view(-1, 1).expand(-1, r_ht.size(2))
        final_idx = final_idx.unsqueeze(1)
        final_rt = r_ht.gather(1, final_idx).squeeze(1)

        author_embeds = []
        if self.config['build_author_track']:
            w_ht = self._bert_encoder(write_track[0], write_track[1] )
            track_read = [r_ht]
            track_write = [w_ht]
            # tracks = [r_ht, w_ht]
            if self.config['build_sentiment_embedding']:
                track_read.extend([self.vader_embed( sentiments[0]),
                               self.flair_embed(sentiments[1]),
                               self.sent_embed(sentiments[2]),
                               self.subj_embed(sentiments[3])])
                track_write.extend([self.vader_embed( sentiments[0]),
                               self.flair_embed(sentiments[1]),
                               self.sent_embed(sentiments[2]),
                               self.subj_embed(sentiments[3])])
                               
            if self.config['leverage_emotion']:
                track_read.append( emotion)
                track_write.append( emotion)

            track_embeds_read = torch.cat(track_read, dim=-1)[:, :-1, :]
            track_embeds_write = torch.cat(track_write, dim=-1)[:, :-1, :]

            track_embeds_read = self.timestamp_merge(track_embeds_read)
            track_embeds_write = self.timestamp_merge(track_embeds_write)

            _, track_ht_read = self._rnn_encode_(self.track_encoder_read, track_embeds_read,
                                            seq_len - 1)
            _, track_ht_write = self._rnn_encode_(self.track_encoder_write, track_embeds_write,
                                            seq_len - 1)
            author_embeds.append(track_ht_read)
            author_embeds.append(track_ht_write)
            
        if self.config['build_author_emb']:            
            author_embeds.append(self.author_embedding(author))

        if len(author_embeds) > 1:
            author_embeds = torch.cat(author_embeds, dim=-1)
        elif len(author_embeds) == 1:
            author_embeds = author_embeds[0]
        else:
            raise NotImplementedError()
        
        if self.config['sentiment_fingerprinting']:
            result['flair'] = self.flair_predict(torch.cat((author_embeds, final_rt), dim=-1))  # batch, seq, 3
            result['vader'] = self.vader_predict(torch.cat((author_embeds, final_rt), dim=-1))
            result['sent'] = self.blob_sent(torch.cat((author_embeds, final_rt), dim=-1))
            result['subj'] = self.blob_subj(torch.cat((author_embeds, final_rt), dim=-1))

        if self.config['emotion_fingerprinting']:
            result['emotion'] = self.emotion_predict(torch.cat((author_embeds, final_rt), dim=-1))

        #convert sentiment dictionary to sentiment tensors
        order = ['vader', 'flair', 'sent', 'subj']
        ls_train_sent = [result[each] for each in order ]
        labels = sentiments
        if self.config['emotion_fingerprinting']:
            ls_train_sent.append( result['emotion']) 
            labels.append( emotion)
        ls_train_sent = torch.stack(ls_train_sent)
        
        return ls_train_sent, labels

    #helper functions    
    def _bert_encoder(self, seq_seq_tensor, token_mask):
        #Here token_mask is attention_mask in bert token
        if len(seq_seq_tensor.size()) == 3:
            batch_size, seq_len, token_size = seq_seq_tensor.size()
        elif len(seq_seq_tensor.size()) == 2:
            batch_size, token_size = seq_seq_tensor.size()
            seq_len = self.seq_len
        else:
            raise NotImplementedError("Not support input dimensions {}".format(x.size()))
        # batch_size, seq_len, token_size = seq_seq_tensor.size()
        token_encode_mtx = seq_seq_tensor.reshape(-1, token_size)
        attention_mtx = token_mask.reshape(-1, token_size)
        token_len = token_mask.reshape(-1, token_size).sum(-1)

        # Feed input to BERT/ELECTRA
        
        model_outputs = self.track_embedder(input_ids= token_encode_mtx, attention_mask= attention_mtx)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = model_outputs[0][:, 0, :]

        cls_format = last_hidden_state_cls.view(batch_size, seq_len, last_hidden_state_cls.size(-1))
        return cls_format

    def _rnn_encode_(self, rnn, x, length, order=None, track=None):
        if len(x.size()) == 3:
            batch_size, seq_len, token_num = x.size()
        elif len(x.size()) == 2:
            batch_size, token_num = x.size()
        else:
            raise NotImplementedError("Not support input dimensions {}".format(x.size()))

        if order is not None:
            x = x.index_select(0, order)
        x = self.dropout(x)
        x = pack(x, length, batch_first=True, enforce_sorted=False)
        
        outputs, h_t = rnn(x)
        outputs = unpack(outputs, batch_first=True)[0]
        if isinstance(h_t, tuple):
            h_t = h_t[0]
        if track is not None:
            outputs = outputs[track]
            h_t = h_t.index_select(1, track).transpose(0, 1).contiguous()
        else:
            h_t = h_t.transpose(0, 1).contiguous()
        return outputs, h_t.view(batch_size, -1)

    def training_step(self, batch, batch_idx):        
        # training_step defined the train loop. It is independent of forward
        batch = self.batch_transform(batch)        
        articles, topics = None, None        
        # author, r_tracks, w_tracks, sentiment, emotion = batch
        if self.config['emotion_fingerprinting']:
            author, r_tracks, w_tracks, sentiment, emotion = batch            
        else:
            author, r_tracks, w_tracks, sentiment = batch
            emotion = np.NaN
        train_result, labels = self(author, r_tracks, w_tracks, articles, sentiment, emotion)#, self.device)        
        #calculate loss
        loss = self.get_loss( train_result, (labels, author), self.train_loss)
        #train accuracy 
        accs = [m(train_result, labels) for m in self.train_accuracy]  # update metric counters
        # Log training loss
        self.log('train_loss', loss, on_step=True)#, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.batch_transform( batch)
        articles, topics = None, None        
        # author, r_tracks, w_tracks, sentiment, emotion = batch
        if self.config['emotion_fingerprinting']:
            author, r_tracks, w_tracks, sentiment, emotion = batch            
        else:
            author, r_tracks, w_tracks, sentiment = batch
            emotion = np.NaN
        val_result, labels = self(author, r_tracks, w_tracks, articles, sentiment, emotion)#, self.device)
        val_loss = self.get_loss( val_result, (labels, author), self.dev_loss)
        #train accuracy 
        accs = [m(val_result, labels) for m in self.val_accuracy]  # update metric counters

        f1_scores = [ ]  # update metric counters
        for i,m in enumerate(self.val_f1):  # update metric counters
            preds, target = self._metric_input_format( i, val_result, labels)
            each_f1_score = m(preds, target)
            f1_scores.append( each_f1_score)
        
        return {"val_loss": val_loss}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)
        avg_acc, avg_f1 = [], []
        for i,m in enumerate(self.val_accuracy):
            self.log('val_acc'+str(i), m.compute())
            avg_acc.append( m.compute())

        for i,m in enumerate(self.val_f1):
            self.log('val_f1'+str(i), m.compute())
            avg_f1.append( m.compute())

        self.log( 'avg_val_acc', sum(avg_acc)/ len(avg_acc))
        self.log( 'avg_val_f1', sum(avg_f1)/ len(avg_f1))

    def test_step(self, batch, batch_idx):
        batch = self.batch_transform( batch)

        articles, topics = None, None        
        # author, r_tracks, w_tracks, sentiment, emotion = batch
        if self.config['emotion_fingerprinting']:
            author, r_tracks, w_tracks, sentiment, emotion = batch            
        else:
            author, r_tracks, w_tracks, sentiment = batch
            emotion = np.NaN
        test_result, labels = self(author, r_tracks, w_tracks, articles, sentiment, emotion)#, self.device)
        loss = self.get_loss( test_result, (labels, author), self.test_loss)

        accs = [m(test_result, labels) for m in self.test_accuracy]  # update metric counters
        f1_scores = [ ]  # update metric counters
        for i,m in enumerate(self.test_f1):  # update metric counters
            preds, target = self._metric_input_format( i, test_result, labels)
            each_f1_score = m(preds, target)
            f1_scores.append( each_f1_score)
        return loss

    def _metric_input_format(self, aspect, preds, target):
        preds = torch.argmax( preds[ aspect], dim=  1)
        target = target[ aspect, :, -1]
        assert preds.shape == target.shape
        return preds, target

    def on_test_epoch_end(self):
        test_result = {}
        
        for i,m in enumerate(self.test_accuracy):
            index = 'test_acc'+str(i)
            print( index+ ' : ' , m.compute())
            test_result[index] = m.compute().detach().cpu().item()

        for i,m in enumerate(self.test_f1):
            index = 'test_f1'+str(i)
            print( index+ ' : ' , m.compute())
            test_result[index] = m.compute().detach().cpu().item()

        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("_%m%d%Y_%H%M")
        dump_file = 'test_perf'+dt_string + '.json' 
        # json.dump(test_result, open(os.path.join( self.config['root_folder'], self.config['outlet'],dump_file ), 'w'))
        analysis_file = os.path.join( self.config['root_folder'], self.config['outlet'],dump_file )
        json.dump(test_result, open( analysis_file, 'w'))
        print('test performance file saved to :', analysis_file)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        optimizer = transformers.AdamW(self.parameters(), lr=self.config['lr']) #, weight_decay=0.01
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                    num_warmup_steps=350,
                                                                                    num_training_steps= 3000,
                                                                                    num_cycles=1)
        schedulers = [    
        {
         'scheduler': scheduler,
         'interval': 'step',
         'frequency': 1
        }]
        return [optimizer], schedulers

    def get_loss(self, train_result, true_result, loss_dict):
        loss = 0
        sentiment, author = true_result
        #sentiment loss
        nsentiment = train_result.shape[0]

        if self.config['emotion_fingerprinting']:
            emotion_loss = F.binary_cross_entropy_with_logits(train_result[-1], sentiment[:, -1] ) #emotion[:,-1])
            loss += emotion_loss
            loss_dict['emotion'].append(emotion_loss.detach().cpu().item())
            nsentiment -= 1 # remove emotion indexing form length

        order = ['vader', 'flair', 'sent', 'subj']
        for i in range(nsentiment):
            each_loss = F.cross_entropy(
                train_result[i],
                sentiment[i][:, -1])
            loss += each_loss
            loss_dict[ order[i] ].append(each_loss.detach().cpu().item())
        return loss

    def batch_transform(self, batch):
        new_batch =[]
        new_batch.append( torch.squeeze( batch[0]) )#author
        new_batch.append(torch.squeeze( batch[1]).transpose(0,1)) #read_track
        new_batch.append(torch.squeeze( batch[2]).transpose(0,1)) #write_track
        new_batch.append(torch.squeeze( batch[3]).transpose(0,1) )#sentiments
        if self.config['emotion_fingerprinting']:
            new_batch.append(torch.squeeze( batch[4])) #emotion
        # batch[0] = torch.squeeze( batch[0]) #author
        # batch[1] = torch.squeeze( batch[1]).transpose(0,1) #read_track
        # batch[2] = torch.squeeze( batch[2]).transpose(0,1) #write_track
        # batch[3] = torch.squeeze( batch[3]).transpose(0,1) #sentiments
        # if self.config['emotion_fingerprinting']:
        #     batch[4] = torch.squeeze( batch[4]) #emotion

        return new_batch  

class AspectACC(pl.metrics.metric.Metric):
    def __init__(self, aspect: int,
                compute_on_step: bool = True,
                dist_sync_on_step: bool = False,
                process_group: Optional[Any] = None,):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,)
        
        self.aspect = aspect
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target) 
        self.total += target.numel()
        
    def compute(self):
        return self.correct.float() / self.total

    def _input_format(self, preds, target):
        preds = torch.argmax( preds[self.aspect], dim=  1)
        target = target[ self.aspect, :, -1]
        return preds, target

