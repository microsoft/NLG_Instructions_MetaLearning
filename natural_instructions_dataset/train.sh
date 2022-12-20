INPUT_DIR=${1:-"/mnt/Data/natural_instructions_v2/tasks"}
OUTPUT_DIR=${2:-"/mnt/Data/"}
CUDA_DEVICE=${3:-0}
EXPERIMENT_NAME=${4:-""}
DATASET_NAME="natural_instructions_v2"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python data/natural_instructions_dataset/train.py \
    --output_dir $OUTPUT_DIR/models/${DATASET_NAME}/${EXPERIMENT_NAME} --overwrite_output_dir \
    --cache_dir $OUTPUT_DIR/model_cache_pt/ --overwrite_cache True \
    --dataset_folder $INPUT_DIR \
    --model_name_or_path facebook/bart-large \
    --tokenizer_name facebook/bart-large \
    --dataset_name="${DATASET_NAME}" \
    --do_train --do_eval --do_predict --predict_with_generate \
    --num_train_epochs 2 \
    --learning_rate 5e-5 --warmup_steps 1 \
    --per_device_train_batch_size 6 --per_device_eval_batch_size 6 \
    --max_source_length 1024 --max_target_length 1024 --decode_max_length 128 \
    --overwrite_output_dir True \
    --train_test_split_mode random_tasks_k_shot \
    --num_train_instances_per_task 1 --num_kshot_train_instances_per_task 1 --num_eval_instances_per_task 1 \
    --metric_for_best_model rouge-l-f --multi_ref_mode best \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 10 --load_best_model_at_end \
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
    --filter_dataset_by_length True \
    --log_task_level_metrics True \
    --use_exclude_tasks_list False \
    --use_train_tasks_list task1540_parsed_pdfs_summarization \
    --use_test_tasks_list  task1572_samsum_summary,task1540_parsed_pdfs_summarization,task1553_cnn_dailymail_summarization \
    --disable_tqdm True --save_checkpoints True  --show_example_predictions 4 \
    --prepend_inst_in_enc_or_dec encoder --mask_label_for_decoder_inst False
