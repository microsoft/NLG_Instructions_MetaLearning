# Project

This repo shares the code for the paper "Boosting Natural Language Generation from Instructions with Meta-Learning", Budhaditya Deb, Guoqing Zheng, Ahmed Hassan Awadallah, EMNLP 2022, https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.456/ 


# Setup

## Setting up the Data Directory
- Download data from repository: allenai/natural-instructions-expansion: Expanding natural instructions. https://github.com/allenai/natural-instructions-expansion
- Run data_utils.py for some basic splits and preprocessing of the NI dataset



## Training and evaluation

Use python 3.9. Install Pytorch 1.9, torchvision 0.10 and torchaudio 0.9 for your cuda version. For example:
  - conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
  - OR, pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  - OR, conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

Install remaining packages from `requirements.txt` file
- transformers
- datasets nltk learn2learn rouge-metric rouge_score


# Example Command Lines for Training (please change the paths according to your installations)

## Standard with BART
CUDA_VISIBLE_DEVICES=2 python data/natural_instructions_dataset/train.py \
--output_dir /mnt/Data/models/natural_instructions_v2.5/standard/6/ --overwrite_output_dir \
--cache_dir /mnt/Data/model_cache_pt/ --overwrite_cache True \
--dataset_folder /mnt/Data/natural_instructions_v2.5/ \
--model_name_or_path facebook/bart-base \
--tokenizer_name facebook/bart-base \
--dataset_name="natural_instructions_v2.5" \
--do_train --do_eval --do_predict --predict_with_generate \
--per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
--max_source_length 1024 --max_target_length 128 --decode_max_length 128 \
--overwrite_output_dir True \
--metric_for_best_model rouge-l-f --multi_ref_mode best \
--adafactor True --fp16 False  --lr_scheduler_type linear  \
--disable_tqdm False \
--add_category False \
--add_task_summary False \
--add_instruction True \
--add_positive_examples True \
--add_negative_examples False \
--add_explanations False \
--num_examples_in_instruction 1 \
--train_loop_opt_type standard \
--filter_dataset_by_length True  \
--disable_tqdm True --save_checkpoints True  --resume_from_checkpoint True --show_example_predictions 0 \
--evaluation_strategy steps --eval_steps 500 --save_steps 500 --logging_steps 50 --load_best_model_at_end True \
--prepend_inst_in_enc_or_dec encoder --mask_label_for_decoder_inst False \
--learning_rate 0.00005 --warmup_steps 500 --gradient_accumulation_steps 1 \
--num_train_epochs 1 \
--train_test_split_mode standard_train_test_dev \
--num_train_instances_per_task 10 --num_kshot_train_instances_per_task 0 \
--num_test_instances_per_task 10 --num_eval_instances_per_task 10 \
--use_train_tasks_list TASK_LIST_GENERATION_V2_5 \
--use_test_tasks_list  False \
--use_eval_tasks_list  TEST_TASKS_LIST_SUMM_TITLE \
--use_exclude_tasks_list  EXCLUDE_TASKS_LIST_SUMM_TITLE \
--log_task_level_metrics True --resume_from_checkpoint True \
--filter_sequences_by_len_diff_from_ref False --pred_ref_diff_tolerance 1.0  --use_train_dataset_for_eval False \
--run_hnet_in_batch_mode_per_task True --task_batch_size_per_iter 3  --num_tasks_per_iter 2  --per_device_train_batch_size 1


# With MAML, BART
CUDA_VISIBLE_DEVICES=0 python data/natural_instructions_dataset/train.py \
--output_dir /mnt/Data/models/natural_instructions_v2.5/maml/3/ --overwrite_output_dir \
--cache_dir /mnt/Data/model_cache_pt/ --overwrite_cache True \
--dataset_folder /mnt/Data/natural_instructions_v2.5/ \
--model_name_or_path facebook/bart-base \
--tokenizer_name facebook/bart-base \
--dataset_name="natural_instructions_v2.5" \
--do_train --do_eval --do_predict --predict_with_generate \
--per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
--max_source_length 1024 --max_target_length 128 --decode_max_length 128 \
--overwrite_output_dir True \
--metric_for_best_model rouge-l-f --multi_ref_mode best \
--adafactor True --fp16 False  --lr_scheduler_type linear  \
--disable_tqdm False \
--add_category False \
--add_task_summary False \
--add_instruction True \
--add_positive_examples True \
--add_negative_examples False \
--add_explanations False \
--num_examples_in_instruction 1 \
--train_loop_opt_type maml \
--task_sampling_mode uniform --target_task_sampling_rate_increase 0.000000 \
--filter_dataset_by_length True \
--disable_tqdm True --save_checkpoints True \
--show_example_predictions 2 \
--learning_rate 0.00001 --warmup_steps 100 --gradient_accumulation_steps 5 \
--inner_loop_learning_rate 0.000005 --use_multi_step_loss_optimization False --use_second_order_gradients False \
--num_inner_training_steps_per_iter 3 --task_batch_size_per_iter 3  --num_tasks_per_iter 2 \
--use_meta_sgd False --meta_sgd_per_param_per_layer False \
--log_task_level_metrics True \
--prepend_inst_in_enc_or_dec encoder \
--log_task_level_metrics True \
--num_train_epochs 20 --evaluation_strategy steps --eval_steps 200 --save_steps 200 --logging_steps 5 --load_best_model_at_end True \
--train_test_split_mode standard_train_test_dev \
--num_train_instances_per_task 10 --num_kshot_train_instances_per_task 0 \
--num_test_instances_per_task 10 --num_eval_instances_per_task 10 \
--use_train_tasks_list TASK_LIST_GENERATION_V2_5 \
--use_test_tasks_list False \
--use_eval_tasks_list  TEST_TASKS_LIST_SUMM_TITLE \
--use_exclude_tasks_list  EXCLUDE_TASKS_LIST_SUMM_TITLE \
--filter_sequences_by_len_diff_from_ref False --pred_ref_diff_tolerance 1.0 --use_train_dataset_for_eval False


# With HNET, BART
CUDA_VISIBLE_DEVICES=3 python data/natural_instructions_dataset/train.py \
--output_dir /mnt/Data/models/natural_instructions_v2.5/hnet/4/ --overwrite_output_dir \
--cache_dir /mnt/Data/model_cache_pt/ --overwrite_cache True \
--dataset_folder /mnt/Data/natural_instructions_v2.5/ \
--model_name_or_path facebook/bart-base \
--tokenizer_name facebook/bart-base \
--dataset_name="natural_instructions_v2.5" \
--do_train --do_eval --do_predict --predict_with_generate \
--per_device_train_batch_size 3 --per_device_eval_batch_size 4 \
--max_source_length 1024 --max_target_length 128 --decode_max_length 128 \
--overwrite_output_dir True \
--train_test_split_mode random_tasks_k_shot \
--metric_for_best_model rouge-l-f --multi_ref_mode best \
--adafactor True --fp16 False  --lr_scheduler_type linear --gradient_checkpointing False \
--disable_tqdm False \
--add_category False \
--add_task_summary False \
--add_instruction True \
--add_positive_examples True \
--add_negative_examples False \
--add_explanations False \
--num_examples_in_instruction 1 \
--filter_dataset_by_length True --log_task_level_metrics True \
--disable_tqdm True --save_checkpoints True  --show_example_predictions 5 \
--prepend_inst_in_enc_or_dec encoder --mask_label_for_decoder_inst False \
--learning_rate 0.000005 --warmup_steps 100 --gradient_accumulation_steps 5 \
--log_task_level_metrics True \
--train_loop_opt_type hnet --hnet_opt_mode alternating_hnet_main --hnet_hidden_dim 128 \
--hnet_exclude_encoder_or_decoder restricted_encoder \
--hnet_add_instruction_to_mainlm True --hnet_use_encoder_last_hidden_state False --hnet_alternating_num_steps 10 \
--hnet_weighet_delta_params False \
--hnet_use_eval_for_nograd True \
--hnet_model_name_or_path facebook/bart-base --hnet_tokenizer_name facebook/bart-base \
--num_train_epochs 5 --evaluation_strategy steps --eval_steps 200 --save_steps 200 --logging_steps 50 --load_best_model_at_end False \
--train_test_split_mode standard_train_test_dev \
--num_train_instances_per_task 10 --num_kshot_train_instances_per_task 0 \
--num_test_instances_per_task 10 --num_eval_instances_per_task 10 \
--use_train_tasks_list TASK_LIST_GENERATION_V2_5 \
--use_test_tasks_list  False \
--use_eval_tasks_list  TEST_TASKS_LIST_SUMM_TITLE \
--use_exclude_tasks_list  EXCLUDE_TASKS_LIST_SUMM_TITLE \
--hnet_include_layer positions \
--run_hnet_in_batch_mode_per_task True --task_batch_size_per_iter 3  --num_tasks_per_iter 2  --per_device_train_batch_size 1

# With HNET_MAML, BART
CUDA_VISIBLE_DEVICES=2 python data/natural_instructions_dataset/train.py \
--output_dir /mnt/Data/models/natural_instructions_v2.5/hnet_maml/ --overwrite_output_dir \
--cache_dir /mnt/Data/model_cache_pt/ --overwrite_cache True \
--dataset_folder /mnt/Data/natural_instructions_v2.5/ \
--model_name_or_path facebook/bart-base \
--tokenizer_name facebook/bart-base \
--dataset_name="natural_instructions_v2.5" \
--do_train --do_eval --do_predict --predict_with_generate \
--max_source_length 1024 --max_target_length 128 --decode_max_length 128 \
--overwrite_output_dir True \
--metric_for_best_model rouge-l-f \
--adafactor True --fp16 False  --lr_scheduler_type linear  \
--disable_tqdm False \
--add_category False \
--add_task_summary False \
--add_instruction True \
--add_positive_examples True \
--add_negative_examples False \
--add_explanations False \
--num_examples_in_instruction 1 \
--prepend_inst_in_enc_or_dec encoder  \
--filter_dataset_by_length True \
--disable_tqdm True --save_checkpoints False \
--show_example_predictions 2 \
--log_task_level_metrics True \
--learning_rate 0.000005 --warmup_steps 100 --gradient_accumulation_steps 1 \
--train_loop_opt_type hnet_maml --hnet_opt_mode alternating_hnet_main --hnet_hidden_dim 128 --hnet_exclude_encoder_or_decoder restricted_encoder \
--hnet_model_name_or_path facebook/bart-base --hnet_tokenizer_name facebook/bart-base \
--hnet_add_instruction_to_mainlm True --hnet_use_encoder_last_hidden_state False --hnet_alternating_num_steps 10 \
--hnet_use_eval_for_nograd True \
--num_inner_training_steps_per_iter 4 --task_batch_size_per_iter 3  --num_tasks_per_iter 2 --use_meta_sgd False \
--inner_loop_learning_rate 0.000001 --task_sampling_mode uniform --target_task_sampling_rate_increase 0.000000 \
--use_multi_step_loss_optimization False --use_second_order_gradients False \
--filter_sequences_by_len_diff_from_ref False --pred_ref_diff_tolerance 1.0  --log_task_level_metrics True \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
--num_train_epochs 5 --evaluation_strategy steps --eval_steps 100 --save_steps 100 --logging_steps 25 --load_best_model_at_end True \
--train_test_split_mode standard_train_test_dev \
--num_train_instances_per_task 10 --num_kshot_train_instances_per_task 0 \
--num_test_instances_per_task 10 --num_eval_instances_per_task 10 \
--use_train_tasks_list TASK_LIST_GENERATION_V2_5 \
--use_test_tasks_list  False \
--use_eval_tasks_list  TEST_TASKS_LIST_SUMM_TITLE \
--use_exclude_tasks_list  EXCLUDE_TASKS_LIST_SUMM_TITLE \
--run_hnet_in_batch_mode_per_task True

## With GPT2
CUDA_VISIBLE_DEVICES=1 python data/natural_instructions_dataset/train.py \
--output_dir /mnt/Data/models/natural_instructions_v2/ --overwrite_output_dir \
--cache_dir /mnt/Data/model_cache_pt/ --overwrite_cache True \
--dataset_folder /mnt/Data/natural_instructions_v2/ \
--model_name_or_path gpt2-medium \
--tokenizer_name gpt2-medium \
--dataset_name="natural_instructions_v2" \
--do_train --do_eval --do_predict --predict_with_generate \
--num_train_epochs 100 \
--learning_rate 5e-5 --warmup_steps 500 \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
--max_source_length 1024 --max_target_length 1024 --decode_max_length 128 \
--overwrite_output_dir True \
--train_test_split_mode random_tasks_k_shot \
--num_train_instances_per_task 1 --num_kshot_train_instances_per_task 1 --num_eval_instances_per_task 5 \
--metric_for_best_model rouge-l-f --multi_ref_mode best \
--evaluation_strategy steps --eval_steps 200 --save_steps 1000 --logging_steps 20 --load_best_model_at_end \
--adafactor True --fp16 False  --lr_scheduler_type linear --gradient_accumulation_steps 10 \
--disable_tqdm False \
--add_category False \
--add_task_summary False \
--add_instruction False \
--add_positive_examples False \
--add_negative_examples False \
--add_explanations False \
--num_examples_in_instruction 1 \
--train_loop_opt_type standard \
--filter_dataset_by_length True --log_task_level_metrics True \
--use_exclude_tasks_list False \
--use_train_tasks_list task1540_parsed_pdfs_summarization --use_test_tasks_list task1540_parsed_pdfs_summarization  \
--disable_tqdm True --save_checkpoints False  --show_example_predictions 5 \
--prepend_inst_in_enc_or_dec encoder --mask_label_for_decoder_inst False

## With LED
CUDA_VISIBLE_DEVICES=0 python data/natural_instructions_dataset/train.py \
--output_dir /mnt/Data/models/natural_instructions_v2/ --overwrite_output_dir \
--cache_dir /mnt/Data/model_cache_pt/ --overwrite_cache True \
--dataset_folder /mnt/Data/natural_instructions_v2/ \
--model_name_or_path allenai/led-large-16384 \
--tokenizer_name allenai/led-large-16384 \
--dataset_name="natural_instructions_v2" \
--do_train --do_eval --do_predict --predict_with_generate \
--num_train_epochs 100 \
--learning_rate 5e-5 --warmup_steps 0 \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
--max_source_length 2048 --max_target_length 2048 --decode_max_length 128 \
--overwrite_output_dir True \
--train_test_split_mode random_tasks_k_shot \
--num_train_instances_per_task 1 --num_kshot_train_instances_per_task 1 --num_eval_instances_per_task 1 \
--metric_for_best_model rouge-l-f --multi_ref_mode best \
--evaluation_strategy steps --eval_steps 200 --save_steps 200 --logging_steps 2000 --load_best_model_at_end \
--adafactor True --fp16 False  --lr_scheduler_type linear --gradient_accumulation_steps 1 \
--disable_tqdm False \
--add_category False \
--add_task_summary False \
--add_instruction True \
--add_positive_examples False \
--add_negative_examples False \
--add_explanations False \
--num_examples_in_instruction 2 \
--train_loop_opt_type standard \
--use_train_tasks_list task1572_samsum_summary  \
--use_test_tasks_list  task1572_samsum_summary,task1540_parsed_pdfs_summarization,task1553_cnn_dailymail_summarization \
--use_exclude_tasks_list False \
--filter_dataset_by_length True --log_task_level_metrics True \
--pred_ref_diff_tolerance 1.0 \
--log_level warning --disable_tqdm True --save_checkpoints False  --show_example_predictions 4 \
--prepend_inst_in_enc_or_dec decoder --mask_label_for_decoder_inst False --gradient_checkpointing True

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## Contributing

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
