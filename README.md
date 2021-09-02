# Opinion Prediction with User fingerprinting
1.	Command line code for Examples creation

    a.	To create examples of train and test datasets,
      i.	python ~/code/fingerprint_lightning/dynamicFPE/dfp_ex_creation_V5.py <author_start_id> <author_end_id> <gpu_id> <Minimum history threshold for author> <Length of relevant history>

    b.	Example: python ~/code/fingerprint_lightning/dynamicFPE/dfp_ex_creation_V5.py -1 -1 0 10 10
      i.	Arguments -1 -1 can be used to refer to all authors

2.	Code for Experiments
  
  a.	Static FPE
  
    i.	DistilBERT
    python ~/code/fingerprint_lightning/dfp_main_V2.py 
    --root_folder ~/data/outlets 
    --batch_size 512 
    --rnn_type gru 
    --grad_clip 0.8 
    --dropout 0.2 
    --epoch 10 
    --previous_comment_cnt 12 
    --freeze_bert True 
    --gpu_id 5 
    --sel_model bert 
    --outlet NewYorkTimes
    --input_examples pickle_inputs/final_mergedexamples_dFP_mh12_rh12_02262021_1428.pkl 
    --model_type staticfpe 
    --history_type static

    ii.	ELECTRA
    python ~/code/fingerprint_lightning/dfp_main_V2.py 
    --root_folder ~/data/outlets 
    --batch_size 512 
    --rnn_type gru 
    --grad_clip 0.8 
    --dropout 0.2 
    --epoch 10 
    --previous_comment_cnt 12 
    --freeze_bert True 
    --gpu_id 5 
    --sel_model bert 
    --outlet theguardian
    --input_examples pickle_inputs/final_mergedexamples_dFP_mh12_rh12_02262021_1428.pkl 
    --model_type staticfpe
    --history_type static

  b.	Dynamic FPE
  
    i.	DistilBERT
    python ~/code/fingerprint_lightning/dfp_main_V2.py 
    --root_folder ~/data/outlets 
    --batch_size 512 
    --rnn_type gru 
    --grad_clip 0.8 
    --dropout 0.2 
    --epoch 10 
    --previous_comment_cnt 15 
    --freeze_bert True 
    --gpu_id 5 
    --sel_model bert 
    --outlet Archiveis 
    --input_examples pickle_inputs/final_mergedexamples_dFP_mh15_rh15_02262021_1428.pkl 
    --model_type dynamicfpe 
    --history_type dynamic

    1.To use static history, static keyword can be used in the history_type argument

    ii.	ELECTRA
    python ~/code/fingerprint_lightning/dfp_main_V2.py 
    --root_folder ~/data/outlets 
    --batch_size 512 
    --rnn_type gru 
    --grad_clip 0.8 
    --dropout 0.2 
    --epoch 10 
    --previous_comment_cnt 5 
    --freeze_bert True 
    --gpu_id 5 
    --sel_model electra 
    --outlet Archiveis 
    --input_examples pickle_inputs/final_mergedexamples_dFP_mh5_rh5_02262021_1428.pkl 
    --model_type dynamicfpe 
    --history_type dynamic

3.	Code for Model analysis
  
  a.	python ~ /code/fingerprint_lightning/model_analysis.py 
  --root_folder ~/data/outlets 
  --batch_size 8 
  --grad_clip 0.8 
  --previous_comment_cnt 12 
  --freeze_bert True 
  --gpu_id 0 
  --sel_model bert 
  --outlet Archiveis 
  --input_examples pickle_inputs/shist_analysis_2871265.pkl 
  --model_type staticfpe 
  --history_type static 
  --load_checkpoint True 
  --path_checkpoint ~/checkpoints/staticfpe-Archiveis-bert-frozTrue-bs128-epoch=08-avg_val_acc=0.5935.ckpt 
  --only_test True

4. Data is available at https://bit.ly/3kLwH3L
