python /disk2/kishore/code/fingerprint_lightning/dfp_main_V2.py 
--root_folder /disk2/kishore/kishore_data/outlets 
--batch_size 128 
--build_author_predict False 
--build_topic_predict False 
--rnn_type gru 
--rnn_layer 2 
--grad_clip 0.8 
--previous_comment_cnt 15 
--freeze_bert True 
--gpu_id 0 
--sel_model electra 
--outlet Archiveis 
--input_examples pickle_inputs/final_mergedexamples_dFP_mh15_rh15_01222021_1143.pkl 
--model_type dynamicfpe 
--history_type dynamic
--dropout 0.2