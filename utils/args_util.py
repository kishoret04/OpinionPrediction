import argparse


def str2bool(v):
    if v.lower() in ['yes', 'y', 1, 'true', 't']:
        return True
    elif v.lower() in ['no', 'n', 0, 'false', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def add_general_args(args=None):
    args = args if args else argparse.ArgumentParser()
    args.add_argument('--random_seed', type=int, default=126)
    # args.add_argument('--gpu_id', type=int, default=1)
    args.add_argument('--gpu_id', type=int, nargs = '+',default= [1])# '--nargs-int-type', nargs='+', type=int
    # args.add_argument('--root_folder', type=str, default='D:/data/outlets')
    # args.add_argument('--root_folder', type=str, default='/home/kishore/Fanyang_code/news/news/outlets')#kishore_update
    args.add_argument('--root_folder', type=str, default= '/home/kishore/kishore_data/outlets') #kishore_update
    args.add_argument('--previous_comment_cnt', type=int, default=12)
    args.add_argument('--min_comment_cnt', type=int, default=14)
    # args.add_argument('--max_seq_len', type=int, default=128)
    args.add_argument('--max_seq_len', type=int, default= 183) #new_kishore_update
    # args.add_argument('--max_title_len', type=int, default= 16)#kishore_update
    args.add_argument('--max_title_len', type=int, default= 12)#new_kishore_update
    args.add_argument('--max_comment_len', type=int, default= 172)#new_kishore_update
    args.add_argument('--prob_to_full', type=float, default=1.)
    args.add_argument('--embedding_weight', default='d:/data/embedding/en.wiki.bpe.vs25000.d300.w2v.txt')
    args.add_argument('--input_examples', default='/disk2/kishore/kishore_data/outlets/NewYorkTimes/pickle_inputs/final_mergedexamples_dFP_mh15_rh15.pkl')
    # args.add_argument('--embedding_weight', default='')
    args.add_argument('--outlet', type=str, default='NewYorkTimes')
    args.add_argument('--model_type', default='staticfpe', choices=['staticfpe', 'dynamicfpe'])
    args.add_argument('--history_type', default='static', choices=['static', 'dynamic'])
    return args

def add_model_args(args=None):
    args = args if args else argparse.ArgumentParser()
    args.add_argument('--rnn_type', default='gru', choices=['lstm', 'gru'])
    # args.add_argument('--hid_dim', type=int, default=256)
    args.add_argument('--hid_dim', type=int, default= 768)#kishore_update size = (768) BERT output
    args.add_argument('--token_dim', type=int, default=300)
    args.add_argument('--dropout', type=float, default=0.)
    args.add_argument('--rnn_layer', type=int, default=2)
    args.add_argument('--rnn_layer_read', type=int, default=2)
    args.add_argument('--rnn_layer_write', type=int, default=1)
    args.add_argument('--author_dim', type=int, default=64)
    args.add_argument('--author_track_dim', type=int, default=256)
    args.add_argument('--topic_dim', type=int, default=64)
    args.add_argument('--emotion_dim', type=int, default=6)
    args.add_argument('--sentiment_dim', type=int, default=64)
    args.add_argument('--build_sentiment_embedding', type=str2bool, default=True)
    args.add_argument('--build_author_emb', type=str2bool, default=False)
    args.add_argument('--build_author_track', type=str2bool, default=True)
    args.add_argument('--build_author_predict', type=str2bool, default=False)
    args.add_argument('--build_topic_predict', type=str2bool, default=False)
    args.add_argument('--leverage_topic', type=str2bool, default=False)
    args.add_argument('--leverage_emotion', type=str2bool, default=False)
    args.add_argument('--sentiment_fingerprinting', type=str2bool, default=True)
    args.add_argument('--emotion_fingerprinting', type=str2bool, default=False)
    args.add_argument('--freeze_bert', type=str2bool, default=True) #kishore_update
    args.add_argument('--sel_model', type=str, default= 'bert', choices = ['bert', 'electra']) #kishore_update
    args.add_argument('--accumulate_grad_batches', type=int, default= 1) #kishore_update
    
    return args


def add_train_args(args=None):
    args = args if args else argparse.ArgumentParser()
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--epoch', type=int, default=10)
    args.add_argument('--update_iter', type=int, default=1)
    args.add_argument('--grad_clip', type=float, default=1.)
    args.add_argument('--use_entire_example_epoch', type=int, default=5)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--update_size', type=int, default=1, help='Backward() without gradient step.')
    args.add_argument('--check_step', type=int, default=100, help='Validate every # steps. ')
    args.add_argument('--load_checkpoint', type=str2bool, default=False)
    args.add_argument('--path_checkpoint', type=str, default='/disk2/kishore')
    args.add_argument('--dataloader_num_workrs', type=int, default=8)#new_kishore_update
    args.add_argument('--only_test', type=str2bool, default=False)
    return args


if __name__ == '__main__':
    """ Test """
    print()