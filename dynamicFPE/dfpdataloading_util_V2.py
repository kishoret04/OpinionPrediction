
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import os
import json
import pytorch_lightning as pl

class DynamicFPDataset(Dataset):
    def __init__(self, output_folder,MAX_LEN_TITLE, MAX_LEN_COMMENT, examples_file):
        #read reference files
        # self.authors = json.load(open(os.path.join(output_folder, 'frequent_author_record.json')))
        self.topic_size = len(open(os.path.join(output_folder, 'vocab.topic')).readlines())

        # article_idx_bert_file =  os.path.join(output_folder, 'shortbert_inputs/article_idx_bert.json');
        # self.df_articles_idx = pd.read_json(article_idx_bert_file, orient='columns') 

        # comment_emosent_bert_file =  os.path.join(output_folder, 'shortbert_inputs/senti_emo_comment_bert.json')
        # self.df_comments = pd.read_json(comment_emosent_bert_file, orient='columns')

        # pure_article_file = os.path.join( output_folder, 'wholebert_inputs/pure_article_bert.json')
        # self.df_pure_article = pd.read_json(pure_article_file, orient='index')        

        article_idx_bert_file =  os.path.join(output_folder, 'pickle_inputs/article_idx_bert.pkl')        
        self.df_articles_idx = pd.read_pickle(article_idx_bert_file) 

        comment_emosent_bert_file =  os.path.join(output_folder, 'pickle_inputs/senti_emo_comment_bert.pkl')
        self.df_comments = pd.read_pickle(comment_emosent_bert_file )
        self.df_comments.index = self.df_comments.index.astype(int)
        
        #create train, val, test data
        self.MAX_LEN_TITLE, self.MAX_LEN_COMMENT = MAX_LEN_TITLE, MAX_LEN_COMMENT 

        #create examples dataset        
        file_examples_pkl = os.path.join(output_folder, examples_file)
        self.df_examples = pd.read_pickle(file_examples_pkl)
        self.authors_ar =  self.df_examples['author'].unique()
        
    def __len__(self):
        return len( self.df_examples)

    def __getitem__(self, idx):
        sr_track = self.df_examples.iloc[idx]
        
        #extract authors
        author = int(sr_track['author'])
        #extract selected comments dataframe
        sel_cid = sr_track.iloc[1:].values
        sel_cid = sel_cid.astype(int) 
        df_comments_sel = self.df_comments.loc[ sel_cid]
        #extract read track 
        sel_aid = df_comments_sel['aid'].values
        df_articles_sel = self.df_articles_idx.loc[ sel_aid]
        ar_art_bert_token = df_articles_sel['t_bert'].tolist()        
        #combine target article with each input article
        read_track = self.extract_comboart_track(ar_art_bert_token)

        #extract write track
        ar_com_bert_token = df_comments_sel['com_bert'].tolist()
        #combine article bert and comment bert along with token type ids and attention masks
        write_track = self.extract_combotrack(ar_art_bert_token,ar_com_bert_token )
        # write_track = self.extract_track(ar_com_bert_token)

        #extract sentiment and emotion
        # emotion_track = df_comments_sel['emotion'].tolist()
        ar_vader = df_comments_sel['vader'].tolist()
        ar_flair = df_comments_sel['flair'].tolist()
        ar_blob_sentiment = df_comments_sel['blob_sentiment'].tolist()
        ar_blob_subjective = df_comments_sel['blob_subjective'].tolist()
        sentiment_track = [ar_vader, ar_flair, ar_blob_sentiment, ar_blob_subjective]
   
        author = torch.tensor([author])
        read_track = torch.tensor([read_track])
        write_track = torch.tensor([write_track])
        sentiment_track = torch.tensor([sentiment_track])
        # emotion_track = torch.tensor([emotion_track], dtype = torch.float)
        
        # return author,read_track, write_track, sentiment_track,emotion_track
        return author,read_track, write_track, sentiment_track

    #split dataset indices
    def datasplit(self, test_train_split=0.9, val_train_split=0.1, shuffle=False ):
        df_dataset = self.df_examples.copy()

        #create test split indices
        df_dataset.reset_index(inplace = True)
        df_result = df_dataset[['author','index']].groupby('author').agg(['count','min','max'])
        df_result.columns = df_result.columns.get_level_values(1)
        df_result['test_split'] = np.floor(df_result['count']*test_train_split).astype(int)
        # print('df_result: \n',df_result)
        #function to create indices for train, val and test sets
        def fun_split_indices(df_in, val_train_split):
            indices = list(range( df_in['min'], df_in['max']+1))
            train_indices, test_indices = indices[: df_in['test_split']], indices[ df_in['test_split']:]
            train_size = len(train_indices)
            validation_split = int(np.floor((1 - val_train_split) * train_size))
            train_indices, val_indices = train_indices[ : validation_split], train_indices[validation_split:]
            df_in['train_indices'] = train_indices
            df_in['val_indices'] = val_indices
            df_in['test_indices'] = test_indices
            return df_in

        df_result = df_result.apply( fun_split_indices,args =(val_train_split,), axis = 1)

        modes = ['train_indices', 'val_indices', 'test_indices']
        self.dic_indices ={}
        # dic_indices ={}
        for each in modes:            
            ls_indices = df_result[each].tolist()
            # print('each: ', ls_indices)
            ls_indices.sort()
            tot_indices = []
            for sub_indices in ls_indices:
                tot_indices.extend( sub_indices) 

            self.dic_indices[each] = tot_indices

        self.train_sampler = SubsetRandomSampler( self.dic_indices['train_indices'])
        self.val_sampler = SubsetRandomSampler( self.dic_indices['val_indices'])
        self.test_sampler = SubsetRandomSampler( self.dic_indices['test_indices'])
        # return dic_indices

    def extract_comboart_track(self,ar_art_bert_token):
        input_id = []
        token_type_id = []
        attention_mask = []
        combo_track = []
        tar_art_bertok = ar_art_bert_token[-1]
        for article in ar_art_bert_token:
            len_art = len(article)
            len_tar = len(tar_art_bertok)
            mod_article = article + [0 for _ in range(self.MAX_LEN_TITLE - len_art)]
            mod_tar = tar_art_bertok + [0 for _ in range(self.MAX_LEN_TITLE - len_tar)]
            combo_input = mod_article + mod_tar[1:]
            combo_attention_mask = [1 if x!=0 else 0 for x in combo_input ]
            combo_tokentype = [0]* self.MAX_LEN_TITLE + [1]* (self.MAX_LEN_TITLE - 1)
            #append tokens
            input_id.append( combo_input)
            attention_mask.append( combo_attention_mask)
            token_type_id.append( combo_tokentype) 

        combo_track.append(input_id)
        combo_track.append( attention_mask)
        combo_track.append( token_type_id)
        return combo_track
    
    def extract_combotrack(self,ar_art_bert_token,ar_com_bert_token ):
        input_id = []        
        attention_mask = []
        token_type_id = []
        combo_track = []
        for article, comment in zip(ar_art_bert_token, ar_com_bert_token):
            len_art = len(article)
            len_com = len(comment)
            mod_article = article + [0 for _ in range(self.MAX_LEN_TITLE - len_art)]
            mod_comment = comment + [0 for _ in range(self.MAX_LEN_COMMENT - len_com)]
            combo_input = mod_article + mod_comment[1:]
            each_attention_mask = [1 if x!=0 else 0 for x in combo_input ]
            each_tokentype = [0]* self.MAX_LEN_TITLE + [1]* (self.MAX_LEN_COMMENT - 1)
            #append tokens
            input_id.append( combo_input)
            attention_mask.append( each_attention_mask)
            token_type_id.append( each_tokentype)

        combo_track.append(input_id)
        combo_track.append( attention_mask)
        combo_track.append( token_type_id)
        return combo_track

    def extract_sentiment_track(self, input_sentiment):
        vader = []
        flair = []
        blob_sentiment = []
        blob_subjective = []
        sentiment_track = []
        {'vader': 2, 'flair': 1, 'blob_sentiment': 1, 'blob_subjective': 2}
        for each in input_sentiment:
            vader.append(each['vader'])
            flair.append(each['flair'])
            blob_sentiment.append(each['blob_sentiment'])
            blob_subjective.append(each['blob_subjective'])
            
        sentiment_track.append(vader)
        sentiment_track.append(flair)
        sentiment_track.append(blob_sentiment)
        sentiment_track.append(blob_subjective)
        return sentiment_track