import os
import json
import pandas as pd
import numpy as np
import sys
import scipy
import torch
from datetime import datetime
import swifter

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

"""
Calculate similarity matrix with sentence transformers for all articles and then accessing it for each example
Enhancing 1st apply about reframing
"""

class CreateSFPExamples:
    def __init__(self, output_folder, device, min_history = -1, relevant_history = 5 ):
        """
        Read reference files
        """
        self.rel_history = relevant_history
        self.device = device
        self.output_folder = output_folder
        self.MINIMUM_HISTORY = min_history
        self.RELEVANT_HISTORY = relevant_history

        file_authors_pkl = os.path.join(output_folder, 'pickle_inputs/frequent_author_record.pkl')
        self.df_authors = pd.read_pickle(file_authors_pkl )
        self.df_authors.index = self.df_authors.index.astype(int)
        
        #reducing data
        if min_history > 0:
            self.df_authors = self.df_authors[ self.df_authors['comments'].apply(len) > min_history]

        self.num_authors = len(self.df_authors)

        print('total number of authors: ', self.num_authors)

    def create_final_ex( self, auth_start_id = -1, auth_end_id= -1):
        COUNT = 1000
        if auth_start_id < 0 and auth_end_id <0 :
            # self.df_authors_selected = self.df_authors
            start, end = 0, self.num_authors
            current_start, current_end = start, start+COUNT
        elif auth_start_id >= 0 and auth_end_id <0 :
            # self.df_authors_selected = self.df_authors.iloc[ auth_start_id:]
            start, end = auth_start_id, self.num_authors
            current_start, current_end = start, start+COUNT
        else:    
            # self.df_authors_selected = self.df_authors.iloc[auth_start_id:auth_end_id]
            # print('authors currently encoded: ', self.df_authors_selected.index.to_list())
            start, end = auth_start_id, auth_end_id
            current_start, current_end = start, end
        
        ####swifter package        
        self.ls_ex_split_filename = []
        while current_end <= end:
            if current_end == end:
                self.df_authors_mod_one = self.df_authors.iloc[current_start: current_end].copy()
                print('index of authors currently encoded are from {0} to {1}. '.format(current_start, current_end))
                current_start = current_end
                current_end = current_end+ 2*COUNT # to end the loop
            elif current_end > end-COUNT:
                self.df_authors_mod_one = self.df_authors.iloc[current_start: ].copy()
                print('index of authors currently encoded are from {0} to {1}. '.format(current_start, end))
                current_start = current_end
                current_end = current_end+ 2*COUNT # to end the loop
            else:
                self.df_authors_mod_one = self.df_authors.iloc[current_start: current_end].copy()
                print('index of authors currently encoded are from {0} to {1}. '.format(current_start, current_end))
                current_start = current_end
                current_end = current_end+COUNT
            
            print('authors currently encoded: ', self.df_authors_mod_one.index.to_list())
            self.df_authors_mod_one = self.df_authors_mod_one.assign( examples = \
                        self.df_authors_mod_one['comments'].swifter.apply(self.fun_reframe)) 
            self.df_authors_mod_one  = self.df_authors_mod_one.swifter.apply(self.fun_expandex, axis = 1)
            self.df_authors_mod_one = self.df_authors_mod_one.swifter.apply(self.fun_test, axis =1 )
            # self.df_authors_mod_one = self.df_authors_mod_one.swifter.apply(self.fun_test, axis =1, args = (self.rel_history,))

            ls_df_final = [pd.DataFrame.from_dict( each, orient = 'index') \
                            for each in (self.df_authors_mod_one['track'].values) if each is not np.NaN]

            self.df_splitex = pd.concat(ls_df_final).reset_index(drop = True)
            now = datetime.now()
            dt_string = now.strftime("_%m%d%Y_%H%M%S")
            examples_file_name = 'pickle_inputs/examples_sFP_mh' + str(self.MINIMUM_HISTORY)+'_relh'+str(self.RELEVANT_HISTORY)+dt_string +'.pkl'
            self.df_splitex.to_pickle( os.path.join( self.output_folder,  examples_file_name) )
            self.ls_ex_split_filename.append( examples_file_name)
            print('completed writing to ', examples_file_name)
            
        self.merge_files( )        
        return self.ls_ex_split_filename

    def merge_files(self):
        #combining all split files
        ls_combined_example_df = [pd.read_pickle( os.path.join( self.output_folder, each )) \
                        for each in self.ls_ex_split_filename]

        df_final = pd.concat( ls_combined_example_df).reset_index(drop = True)
        
        now = datetime.now()
        dt_string = now.strftime("_%m%d%Y_%H%M")
        final_filename = 'pickle_inputs/final_mergedexamples_sFP_mh'+str(self.MINIMUM_HISTORY)+'_rh'+str(self.RELEVANT_HISTORY)+dt_string +'.pkl'
        df_final.to_pickle( os.path.join( self.output_folder,  final_filename) )
        print('Merged all files to : ', final_filename)
        self.ls_ex_split_filename.append(final_filename)
        
    def fun_reframe(self, history):
        #apply function
        # dict_history = {key : [history[ :i+1]] for key, i in enumerate( range( self.rel_history, len(history))) }
        dict_history = {key : history[max(-self.rel_history + i, 0): i + 1]
            for key, i in enumerate(range( self.rel_history, len(history)))}
        # df_history_dict = pd.DataFrame.from_dict(dict_history, orient='index', columns = ['history_track'])
        # df_history_dict = df_history_dict.assign( relevant =  df_history_dict.apply( self.new_rankedhistory,axis = 1  ))
        
        # result = dict(zip(df_history_dict.index.to_list(), df_history_dict['relevant']))
        return dict_history     

    def fun_expandex(self, df_input):
        new_dict = {df_input.name: df_input['examples']}
        df_input['new_example'] = new_dict
        return df_input

    def fun_test(self, df_input):
        prev_comments = self.rel_history
        df_nested  = pd.DataFrame(df_input['new_example'])
        if (len(df_nested)> 0):
            author = str(df_input.name)
            comment_columns = ['t-'+str( prev_comments-i) for i in range(0,prev_comments)]+ ['t']
            df_examples_new = pd.DataFrame(columns= comment_columns )
            df_examples_new[ comment_columns] = pd.DataFrame(df_nested[df_input.name].tolist(), index= df_nested.index)
            df_examples_new.insert(0,'author', author)
            df_input['track'] = df_examples_new.to_dict(orient = 'index')
        return df_input

def main():
    #read arguments
    args = sys.argv[1:]
    # args = [0, 5, 0 ,12,12]
    if args:
        auth_start_id = int(args[0])
        auth_end_id = int(args[1])
        gpu_id = int(args[2])
        #constants
        MINIMUM_HISTORY = int(args[3])
        RELEVANT_HISTORY = int(args[4])
    else:
        print('No author indices given.Taking all authors')
        auth_start_id = -1
        auth_end_id = -1
        gpu_id = 1
        #constants
        MINIMUM_HISTORY = 12
        RELEVANT_HISTORY = 12

    #Changing filepaths
    print('current directory: ', os.getcwd() )
    
    ## NEW LAB PATH
    output_folder_orig = '/disk2/kishore/kishore_data/outlets'#'/home/kishore/kishore_data/outlets'
    data_folder_orig =  '/disk2/kishore/fan_backup/Old_code/news/outlets'#'/home/kishore/fan_backup/Old_code/news/outlets'

    outlet_list = [ 'DailyMail' ,'foxnews', 'theguardian', 'wsj','Archiveis']#, 'NewYorkTimes']
    for outlet in outlet_list:
        # outlet = 'NewYorkTimes'
        print('preprocessing ', outlet)
        output_folder = os.path.join( output_folder_orig, outlet)
        data_folder = os.path.join( data_folder_orig, outlet)
        os.chdir(output_folder)
        print('New directory: ', os.getcwd() )

        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format( gpu_id))#args.gpu_id))       
        else:
            device = torch.device('cpu')
        
        print('device : ', device)

        print('creating examples for authors from {0} to {1}(excluding).'.format(auth_start_id, auth_end_id))

        print('MINIMUM_HISTORY: ', MINIMUM_HISTORY)
        print('RELEVANT_HISTORY: ', RELEVANT_HISTORY)

        outlet_examples = CreateSFPExamples( output_folder, min_history = MINIMUM_HISTORY , 
                        relevant_history = RELEVANT_HISTORY,  device = device)
        
        ls_example_filenames = outlet_examples.create_final_ex(auth_start_id, auth_end_id)
        print('completed creation of examples files with MINIMUM_HISTORY {0} and RELEVANT_HISTORY {1}.'.format(MINIMUM_HISTORY, RELEVANT_HISTORY))
        print('Following are the list of filenames for outlet {0}:\n {1} '.format( outlet, ls_example_filenames))    
    
if __name__ == '__main__':
    main()


