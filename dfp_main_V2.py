import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
import time
import torch
import random
import numpy as np
import pandas as pd
from utils import args_util, plmodel_util,dataloading_V2
from dynamicFPE import dfpdataloading_util_V2, dfpmodel_util
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only
from pl_bolts.callbacks import PrintTableMetricsCallback

import wandb


@rank_zero_only
def wandb_save(wandb_logger, config):
    wandb_logger.log_hyperparams(config)
    wandb_logger.experiment.save('./dfp_main.py', policy="now")

def get_outlet_metrics(given_outlet):
    outlet_list = [ 'foxnews', 'theguardian', 'wsj','Archiveis','NewYorkTimes','DailyMail']
    switcher={
            outlet_list[0]: ( 14, 44),
            outlet_list[1]: ( 14, 103),
            outlet_list[2]: ( 12, 119),
            outlet_list[3]:( 13, 175),
            outlet_list[4]:( 12, 172),
            outlet_list[5]:( 21, 57)
    }
    return switcher.get(given_outlet,"Invalid outlet")
    
def main():
    print( 'current path: ', os.getcwd())
    print(' changing default directory to ', '/disk2/~/home')
    os.chdir('/disk2/~/home')
    print( 'current path: ', os.getcwd())
    wandb.init(project="fp_lightning")

    arg_parser = args_util.add_general_args()
    arg_parser = args_util.add_train_args(arg_parser)
    arg_parser = args_util.add_model_args(arg_parser)
    args = arg_parser.parse_args()

    #~ update parameters
    args_dict = vars(args)
    # args_dict['sel_model'] = 'electra'
    # pl.seed_everything(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    config = {'root_folder': args.root_folder,
              'author_dim': args.author_dim,
              'author_track_dim': args.author_track_dim,
              'topic_dim': args.topic_dim,
              'token_dim': args.token_dim,
              'rnn_type': args.rnn_type,
              'rnn_layer': args.rnn_layer,
              'hid_dim': args.hid_dim,
              'dropout': args.dropout,
              'sentiment_dim': args.sentiment_dim,
              'emotion_dim': args.emotion_dim,
              'build_sentiment_embedding': args.build_sentiment_embedding,
              'build_author_emb': args.build_author_emb,
              'build_author_track': args.build_author_track,
              'build_author_predict': args.build_author_predict,
              'build_topic_predict': args.build_topic_predict,
              'leverage_topic': args.leverage_topic,
              'leverage_emotion': args.leverage_emotion,
              'lr': args.lr,
              'epoch': args.epoch,
              'update_iter': args.update_iter,
              'grad_clip': args.grad_clip,
              'use_entire_example_epoch': args.use_entire_example_epoch,
              'batch_size': args.batch_size,
              'update_size': args.update_size,
              'check_step': args.check_step,
              'random_seed': args.random_seed,
              'previous_comment_cnt': args.previous_comment_cnt,
              'min_comment_cnt': args.min_comment_cnt,
              'max_seq_len': args.max_seq_len,
              'max_title_len': args.max_title_len, #~_update
              'max_comment_len': args.max_comment_len, #~_update
              'prob_to_full': args.prob_to_full,
              'sentiment_fingerprinting': args.sentiment_fingerprinting,
              'emotion_fingerprinting': args.emotion_fingerprinting,
              'freeze_bert' : args.freeze_bert,
              'dataloader_num_workrs': args.dataloader_num_workrs,
              'gpu_id': args.gpu_id,
              'sel_model': args.sel_model,
              'accumulate_grad_batches': args.accumulate_grad_batches,
              'load_checkpoint': args.load_checkpoint,
              'path_checkpoint': args.path_checkpoint,
              'rnn_layer_read': args.rnn_layer_read,
              'rnn_layer_write': args.rnn_layer_write,
              'input_examples': args.input_examples,
              'outlet': args.outlet,
              'model_type': args.model_type,
              'history_type': args.history_type,
              'only_test': args.only_test
              }
    
    ########setting argument config#######
    # python /disk2/~/code/fingerprint_lightning/dfp_main_V2.py --root_folder /disk2/~/~_data/outlets 
    # --batch_size 32 --grad_clip 0.8 --previous_comment_cnt 12 --freeze_bert False --gpu_id 0 --sel_model bert 
    # --outlet Archiveis --input_examples pickle_inputs/shist_analysis_2871265.pkl --model_type staticfpe 
    # --history_type static --load_checkpoint True --path_checkpoint 
    # /disk2/~/home/wandb/run-20210121_173938-3doo9bau/files/fp_lightning/3doo9bau/checkpoints/staticfpe-Archiveis-bert-frozTrue-bs128-epoch=08-avg_val_acc=0.5935.ckpt
    #  --only_test True
    # config['root_folder'] ='/disk2/~/~_data/outlets' 
    # config['batch_size'] = 19
    # config['rnn_type'] = 'gru' 
    # config['dropout'] = 0.2 
    # config['grad_clip'] = 0.8
    # config['previous_comment_cnt'] = 12
    # config['freeze_bert'] = True
    # config['gpu_id'] = [0]
    # config['epoch'] = 20 
    # config['load_checkpoint'] = True
    # config['path_checkpoint'] = '/disk2/~/home/wandb/run-20210121_173938-3doo9bau/files/fp_lightning/3doo9bau/checkpoints/staticfpe-Archiveis-bert-frozTrue-bs128-epoch=08-avg_val_acc=0.5935.ckpt'
    # config['sel_model'] = 'bert'
    # config['input_examples'] ='pickle_inputs/shist_analysis_2871265.pkl'
    # config['outlet'] = 'Archiveis'
    # config['model_type'] = 'staticfpe'
    # config['history_type'] = 'static'
    # config['only_test'] = True
    # # config['dataloader_num_workrs'] = 1
    ######################
    #settings config params#
    if config['sel_model'] == 'electra':
        config['hid_dim'] = 256           

    config['max_title_len'], config['max_comment_len'] = get_outlet_metrics( config['outlet'])
    ######################

    for key, value in config.items():
        print(key, value)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    #loggers
    wandb_name = 'dfpmodel_'+config['sel_model']
    wandb_logger = pl.loggers.WandbLogger(project="fp_lightning", name= wandb_name, log_model= True )
    wandb_save(wandb_logger, config)
    outlet = config['outlet']
    # outlet_list = [ 'DailyMail' ,'foxnews', 'theguardian', 'wsj','Archiveis']#, 'NewYorkTimes']
    # for outlet in [ 'NewYorkTimes'] : #,'Archiveis', 'wsj',]:  # os.listdir(args.root_folder): #~_update
    print("Working on {} ...".format(outlet))
    output_folder = os.path.join( config['root_folder'], outlet)    
    input_examples_pkl = os.path.join(output_folder, config['input_examples'])
    #Dataset creation
    if config['history_type'] == 'static':
        ####### Static fingerprint#######
        input_dataset = dataloading_V2.FinalDataset( output_folder = output_folder,
                                                  MAX_LEN_TITLE = config['max_title_len'],
                                                  MAX_LEN_COMMENT = config['max_comment_len'],
                                                  examples_file = input_examples_pkl)
    else:
        ##############Dynamic FP########                
        input_dataset = dfpdataloading_util_V2.DynamicFPDataset( output_folder = output_folder,
                                                            MAX_LEN_TITLE = config['max_title_len'], 
                                                            MAX_LEN_COMMENT = config['max_comment_len'],
                                                            examples_file = input_examples_pkl)
    #split data to train-test-validation
    input_dataset.datasplit()
    #######################
    config['author_size'] = len(input_dataset.authors_ar)
    config['topic_size'] = input_dataset.topic_size
    # config['outlet'] = outlet

    #create dataloaders
    train_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                sampler= input_dataset.train_sampler, num_workers= config['dataloader_num_workrs'], pin_memory=True)

    val_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                sampler= input_dataset.val_sampler, num_workers= config['dataloader_num_workrs'], pin_memory=True)

    test_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                sampler= input_dataset.test_sampler, num_workers= config['dataloader_num_workrs'], pin_memory=True )

    # init model
    ###Frozen BERT/ELECTRA model
    # fpmodel = plmodel_util.FingerprintModel(config= config)
    # wandb_logger.watch( fpmodel, log='all', log_freq=100) # log = 'gradients'

    ###BERT centric model
    ckp_name = config['model_type']+'-'+ config['outlet']+'-'+ config['sel_model']+'-froz'+ str(config['freeze_bert'])+'-bs'+ str(config['batch_size'])
    if config['model_type'] == 'staticfpe':    
        if config['load_checkpoint']:
            fp_model = plmodel_util.BertFPModel.load_from_checkpoint( checkpoint_path = config['path_checkpoint'])
        else:
            fp_model = plmodel_util.BertFPModel(config= config)
    else:
        if config['load_checkpoint']:
            fp_model = dfpmodel_util.DynamicFPModel.load_from_checkpoint( checkpoint_path = config['path_checkpoint'])
        else:
            fp_model = dfpmodel_util.DynamicFPModel(config= config)

    wandb_logger.watch( fp_model, log='all', log_freq=100) # log = 'gradients'

    #callbacks
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping( monitor='avg_val_acc', min_delta=0.00, 
    patience=3, verbose=False, mode='max')

    # ckp_name = 'fp-'+ config['sel_model']+'-froz'+ str(config['freeze_bert'])+'-bs'+ str(config['batch_size'])
    chk_callback_valacc = ModelCheckpoint( filename=ckp_name + '-{epoch:02d}-{avg_val_acc:.4f}', 
    # dirpath= os.path.join(wandb.run.dir,'checkpoints') ,
    save_top_k=2, monitor='avg_val_acc', mode='max', )

    print_callback = PrintTableMetricsCallback()

    # #debug an epoch run
    # trainer = pl.Trainer( logger = wandb_logger, log_every_n_steps=1, gradient_clip_val = config['grad_clip'], min_epochs = 5, max_epochs = config['epoch'], 
    # val_check_interval = 0.005, callbacks=[early_stop_callback], checkpoint_callback = cp_valacc, auto_scale_batch_size='binsearch', profiler = True, limit_train_batches = 0.7, 
    # accelerator='ddp', plugins='ddp_sharded', gpus = [config['gpu_id']] ,fast_dev_run = True,replace_sampler_ddp=False)  #limit_val_batches=500,

    #######Automatically overfit the same batch of your model for a sanity test
    # trainer = pl.Trainer(overfit_batches=100, logger = wandb_logger, max_epochs = config['epoch'],
    # callbacks = [chk_callback_valacc, print_callback],gradient_clip_val = config['grad_clip'], profiler = "simple", 
    # accelerator='ddp',gpus = config['gpu_id'] ,replace_sampler_ddp=False, 
    # accumulate_grad_batches = config['accumulate_grad_batches'])

    # trainer.tune( fpmodel, train_dataloader= train_loader, val_dataloaders = val_loader) # need to add model.batch_size and change dataloader parameter
    
    # # unit test all the code- hits every line of your code once to see if you have bugs, # instead of waiting hours to crash on validation
    # trainer = pl.Trainer(fast_dev_run=True)
    ############Model training ##########
    #debugging
    # use only 10 train batches and 3 val batches
    # trainer = pl.Trainer( logger = wandb_logger, log_every_n_steps=1, 
    # max_epochs = 3, 
    # limit_train_batches=5, limit_val_batches=1, limit_test_batches =1,
    # checkpoint_callback = True, 
    # callbacks = [chk_callback_valacc, print_callback,early_stop_callback], 
    # gradient_clip_val = config['grad_clip'], profiler = "simple",
    # accelerator= 'ddp', gpus = config['gpu_id'] ,replace_sampler_ddp=False,#plugins='ddp_sharded'
    #          )

    trainer = pl.Trainer( logger = wandb_logger, log_every_n_steps=1, 
    max_epochs = config['epoch'], 
    val_check_interval = 0.1,limit_val_batches=0.1,
    callbacks = [chk_callback_valacc, print_callback],gradient_clip_val = config['grad_clip'], profiler = "simple",
    accelerator='ddp',gpus = config['gpu_id'] ,replace_sampler_ddp=False,
    accumulate_grad_batches = config['accumulate_grad_batches'],
        )  #

    #fitting
    if not config['only_test']:
        trainer.fit( model = fp_model, train_dataloader= train_loader, val_dataloaders= val_loader)
        #test
        trainer.test( model= fp_model, test_dataloaders=test_loader, ckpt_path='best', verbose=True, )
    else:
        test_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                 num_workers= config['dataloader_num_workrs'], pin_memory=True )
        trainer.test( model= fp_model, test_dataloaders=test_loader, verbose=True, )

if __name__ == '__main__':
    main()