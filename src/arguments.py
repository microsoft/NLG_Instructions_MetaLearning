import os
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers.file_utils import add_start_docstrings


# TODO: these should be different for pt and jax
# MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# MODEL_CONFIG_CLASSES_FLAX = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class DerivedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    # ---------------- metrics parans --------------
    log_task_level_metrics: bool = field(default=True, metadata={
        "help": "Whether to use log metrics for each task (ROUGE, BLEU)."})
    filter_sequences_by_len_diff_from_ref: bool = field(default=False, metadata={
        "help": "Whether to filter out sequences too different in length from the reference"})
    pred_ref_diff_tolerance:float = field(default=1.0, metadata={
        "help": "If difference between prediction and reference strings (normalized by reference "
                "lengths) is more than this amount do not consider the metrics"})
    # ---------------- optimization params --------------
    mask_label_for_decoder_inst: bool = field(default=False, metadata={
        "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."})
    multi_ref_mode: str = field(default="average", metadata={
        "help": "For multiple reference, how to compute ROUGE. options: best, average"})
    num_cycles_cosine_schd: Optional[int] = field(default=1, metadata={
        "help": "number of restarts in the cosine scheduler"},)
    power_polynomial_schd: Optional[float] = field(default=1, metadata={
        "help": "number of restarts in the cosine scheduler"},)
    predict_with_generate: bool = field(default=False, metadata={
        "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."})
    prepend_inst_in_enc_or_dec: str = field(default="encoder", metadata={
        "help": "Where to put the instruction, as a prefix for encoder or decoder"})
    save_checkpoints: bool = field(default=False, metadata={
        "help": "By defult do not save checkpoints as it is causing OS errors."})
    show_example_predictions: Optional[int] = field(default=0, metadata={
        "help": "Show some expample predictions for different tasks"},)
    target_task_sampling_rate_increase:float = field(default=0.0, metadata={
        "help": "Linearly increase the rate of target tasks sampling at this rate"})
    train_loop_opt_type: str = field(default="standard", metadata={
        "help": "Type of training loop. Options: standard, maml, hnet"})
    # ------------- meta params ----------------
    enable_inner_loop_optimizable_bn_params: bool = field(default=False,metadata={
        "help": "Enable inner loop BN optimization in MAML"},)
    first_order_to_second_order_steps:int = field(default=100000, metadata={
        "help": "Number of steps after which 2nd order gradients are used in MAML"})
    gradient_checkpointing: bool = field(default=False,metadata={
        "help": "Whether to use gradient checkpointing"},)
    inner_loop_learning_rate:float = field(default=0.00005, metadata={
        "help": "Learning rate for innertask level loop of MAML"})
    max_grad_norm_meta:float = field(default=0.1, metadata={
        "help": "Learning rate for innertask level loop of MAML"})
    multi_step_loss_num_steps: Optional[int] = field(default=100000, metadata={
        "help": "Number of steps multi-step loss in MAML"},)
    num_inner_training_steps_per_iter: Optional[int] = field(default=5, metadata={
        "help": "Number of steps in inner loop of MAML"},)
    use_learnable_per_layer_per_step_inner_loop_learning_rate:bool = field(default=False, metadata={
        "help": "Learning rate for inner loop of MAML"})
    use_multi_step_loss_optimization: bool = field(default=False,metadata={
        "help": "Whether to use multi-stage optimization of MAML++ "},)
    use_second_order_gradients: bool = field(default=True,metadata={
        "help": "Whether to use 2nd order gradients in MAML"},)
    use_meta_sgd: bool = field(default=False,metadata={
        "help": "Whether to use MetaSGD with MAML"},)
    meta_sgd_per_param_per_layer: bool = field(default=False,metadata={
        "help": "Whether to use per parameter per module learning rate in MetaSGD. When set to false it creates one learning rate per layer."},)
    # ------------ hnet params -----------------
    hnet_opt_mode: Optional[str] = field(default="hnet", metadata={
        "help": "The train mode for hnet with main model: hnet, hnet_main, alternating_hnet_main"})
    hnet_alternating_num_steps:int = field(default=1, metadata={
        "help": "Number of steps before switing the training model in alternating HNET"})
    run_hnet_in_batch_mode_per_task: bool = field(default=False,metadata={
        "help": "Whether to use batch mode for hnet, valid for MAML since it samples by tasks"},)
    hnet_initial_alt_mode: Optional[str] = field(default="hnet", metadata={
        "help": "In alternating training mode, whether to start with hnet or main_lm training"})
@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: str = field(metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    cache_dir: Optional[str] = field(default=None,metadata={
        "help": "Where to store the pretrained models downloaded from huggingface.co"},)
    config_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained config name or path if not the same as model_name"})
    dtype: Optional[str] = field(default="float32", metadata={
        "help": "Floating-point format in which the model weights should be initialized "
                "and trained. Choose one of `[float32, float16, bfloat16]`."},)
    model_revision: str = field(default="main", metadata={
        "help": "The specific model version to use (can be a branch name, tag name or commit id)."},)
    prefix_dropout: Optional[float] = field(default=0.0, metadata={
        "help": "Lenght of prefix sequences to feed into the LM"},)
    prefix_lm_mid_dim: Optional[int] = field(default=512, metadata={
        "help": "Dimension of the internal hidden layer of prefix_lm layers"},)
    prefix_seq_len: Optional[int] = field(default=10, metadata={
        "help": "Lenght of prefix sequences to feed into the LM"},)
    tokenizer_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained tokenizer name or path if not the same as model_name"})
    use_auth_token: bool = field(default=False,metadata={
        "help": "Will use the token generated when running `transformers-cli login` "
                "(necessary to use this script with private models)."},)
    use_fast_tokenizer: bool = field(default=True, metadata={
        "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},)
    # ------------- hnet params ----------------
    hnet_config_name: Optional[str] = field(default=None, metadata={
        "help": "For HyperNet: Pretrained config name or path if not the same as model_name"})
    hnet_model_name_or_path: Optional[str] = field(default=None, metadata={
        "help": "For HyperNet: Path to pretrained model or model identifier from huggingface.co/models"})
    hnet_tokenizer_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained tokenizer name or path if not the same as model_name"})
    hnet_max_scale: Optional[int] = field(default=1, metadata={
        "help": "Max scale value for Hnet outputs"},)
    hnet_hidden_dim: Optional[int] = field(default=512, metadata={
        "help": "Max scale value for Hnet outputs"},)
    hnet_exclude_encoder_or_decoder: str = field(default="restricted_encoder",metadata={
        "help": "Exclude parameters for encoder or decoder or both, 'none', 'encoder', 'decoder' "},)
    hnet_include_layer: str = field(default="none",metadata={
        "help": "Include any specific layer such as embedding, position etc."},)
    hnet_use_encoder_last_hidden_state: bool = field(default=False,metadata={
        "help": "Whether to use the last hidden state of encoder or all all hidden states of the decoder for params"},)
    hnet_weighet_delta_params: bool = field(default=False,metadata={
        "help": "Whether learn the weight associated with the delta params from the HNET"},)
    hnet_params_multiply: bool = field(default=False,metadata={
        "help": "Whether add or multiply the weights o from the HNET"},)
    hnet_use_eval_for_nograd: bool = field(default=False,metadata={
        "help": "Whether to use eval or no_grad for freezing layers in HNET and mainlm"},)
    
    # ------------- generation params ----------------
    decode_mode: Optional[str] = field(default="beam", metadata={
        "help": "Different decoding schemes, 'nucleus', 'beam', 'greedy' "},)
    decode_num_return_sequences: Optional[int] = field(default=1, metadata={
        "help": "number of sequences returned during decoding"},)
    decode_length_penalty: Optional[float] = field(default=2.0, metadata={
        "help": "repetition penalty during decoding"},)
    decode_max_length: Optional[int] = field(default=128,metadata={
        "help": "repetition penalty during decoding"},)
    decode_num_beams: Optional[int] = field(default=1, metadata={
        "help": "Beam width used in beam search decoding"},)
    decode_repetition_penalty: Optional[float] = field(default=1.0, metadata={
        "help": "repetition penalty during decoding"},)
    decode_temp: Optional[float] = field(default=1.0, metadata={
        "help": "Decoding temperature for nucleaus sampling"},)
    decode_top_p: Optional[float] = field(default=0.9, metadata={
        "help": "Top probability for filtering during decoding"},)
    decode_topk: Optional[int] = field(default=1, metadata={
        "help": "Topk for Decoding temperature for nucleaus sampling"},)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_config_name: Optional[str] = field(default=None, metadata={
        "help": "The configuration name of the dataset to use (via the datasets library)."})
    dataset_folder: str = field(default=None, metadata={
        "help": "path for the dataset folder"},)
    dataset_name: Optional[str] = field(default=None, metadata={
        "help": "The name of the dataset to use (via the datasets library)."})
    evaluate_metrics_after_epoch: Optional[bool] = field(default=False, metadata={
        "help": "Evaluates metrics at the end of the epoch and saves the best model."},)
    overwrite_cache: bool = field(default=False, metadata={
        "help": "Overwrite the cached training and evaluation sets"})
    summary_column: Optional[str] = field(default=None, metadata={
        "help": "The name of the column in the datasets containing the summaries (for summarization)."},)
    test_file: Optional[str] = field(default=None, metadata={
        "help": "An optional input test data file to evaluate the metrics (rouge) on "
                "(a jsonlines or csv file)."},)
    text_column: Optional[str] = field(default=None,metadata={
        "help": "The name of the column in the datasets containing the full texts (for summarization)."},)
    train_file: Optional[str] = field(default=None, metadata={
        "help": "The input training data file (a jsonlines or csv file)."})
    validation_file: Optional[str] = field(default=None, metadata={
        "help": "An optional input evaluation data file to evaluate the metrics (rouge) "
                "on (a jsonlines or csv file)."},)
    use_train_dataset_for_eval: Optional[bool] = field(default=False, metadata={
        "help": "For zero shot we can use the train datasetitself to test metrics"},)
    # ------------------ data params ------------------
    filter_dataset_by_length: bool = field(default=True, metadata={
        "help": "Set flag to filter data for max sequence lenghts before training starts"})
    ignore_pad_token_for_loss: bool = field(default=True,metadata={
        "help": "Whether to ignore the tokens corresponding to padded labels in the loss "
                "computation or not."},)
    ignore_tasks_list_in_test: str = field(default=None, metadata={
        "help": "List of task categories to ignore in test and validation. E.g:  "
                "'['Classification','Translation']'  "},)
    max_eval_samples: Optional[int] = field(default=None, metadata={
        "help": "For debugging purposes or quicker training, truncate the number of evaluation "
                "examples to this value if set."},)
    max_predict_samples: Optional[int] = field(default=None, metadata={
        "help": "For debugging purposes or quicker training, truncate the number of prediction "
                "examples to this value if set."},)
    max_train_samples: Optional[int] = field(default=None, metadata={
        "help": "For debugging purposes or quicker training, truncate the number of training "
                "examples to this value if set."},)
    max_source_length: Optional[int] = field(default=1024, metadata={
        "help": "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."},)
    max_target_length: Optional[int] = field(default=128, metadata={
        "help": "The maximum total sequence length for target text after tokenization. Sequences "
                "longer than this will be truncated, sequences shorter will be padded."},)
    num_eval_instances_per_task: float = field(default=1.0, metadata={
        "help": "Number of instances of input/output pairs for each class used for evaluation at the end of training. "
                "-1 denotes all available are used."},)
    num_test_instances_per_task: float = field(default=1.0, metadata={
        "help": "Number of instances of input/output pairs for each class used for testing. "
                "-1 denotes all available are used."},)
    num_kshot_train_instances_per_task: float = field(default=0.0, metadata={
        "help": "Number of instances of input/output pairs for each class used for training. "
                "This will be used in k_shot and zero_shot setting. Note the values here sets "
                "the num instances for low resource train tasks (which are also in test tasks). "
                "The values can be set either as a fraction or exact number."},)
    num_train_instances_per_task: float = field(default=1.0, metadata={
        "help": "Number of instances of input/output pairs for each class used for training. "
                "If a fraction <1 is given, it selects instances proportionate to the number "
                "of total instances."},)
    pad_to_max_length: bool = field(default=False, metadata={
        "help": "Whether to pad all samples to model maximum sentence length. If False, will pad "
                "the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."},)
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={
        "help": "The number of processes to use for the preprocessing."},)
    source_prefix: Optional[str] = field(default="[Source]", metadata={
        "help": "A prefix to add before every source text (useful for T5 models)."})
    target_prefix: Optional[str] = field(default="[Target]", metadata={
        "help": "A prefix to add before every target text (useful for T5 models, and for "
                "prompting generation)."})
    val_max_target_length: Optional[int] = field(default=128, metadata={
        "help": "The maximum total sequence length for validation target text after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded. "
                "Will default to `max_target_length`. This argument is also used to override the "
                "``max_length`` param of ``model.generate``, which is used during ``evaluate`` "
                "and ``predict``."},)
    use_exclude_tasks_list: str = field(default="False",metadata={
        "help": "You can pass a list of tasks through a whitelist which would be excuded from "
                "training. This flags whether to use the whitelist or use a randomly split set "
                "of tasks into train or test"},)
    use_train_tasks_list: str = field(default="False",metadata={
        "help": "You can pass a list of test tasks through a whitelist. This flags whether "
                "to use the whitelist or use a randomly split set of tasks into train or test"},)
    use_test_tasks_list: str = field(default="False",metadata={
        "help": "You can pass a list of test tasks through a whitelist. This flags whether "
                "to use the whitelist or use a randomly split set of tasks into train or test"},)
    use_eval_tasks_list: str = field(default="False",metadata={
        "help": "You can pass a list of validation tasks through a whitelist. This flags whether "
                "to use the whitelist or use a randomly split set of tasks into train or test"},)
    # ------------------ intructions params ------------------
    add_category: bool = field(default=False, metadata={
        "help": "Whether to add the taskcategory in front of the input."},)
    add_explanations: bool = field(default=False, metadata={
        "help": "Whether to add explanations to positive/negative examples in the instruction of the input."},)
    add_instruction: bool = field(default=False, metadata={
        "help": "Whether to use the instruction in front of the input."},)
    add_negative_examples: bool = field(default=False, metadata={
        "help": "Whether to add negative examples in the instruction of the input."},)
    add_positive_examples: bool = field(default=False, metadata={
        "help": "Whether to add positive examples in the instruction of the input."},)
    add_task_category: bool = field(default=False, metadata={
        "help": "Whether to use the task summary description with task  in front of the input."},)
    add_task_summary: bool = field(default=False, metadata={
        "help": "Whether to use the task summary description with task  in front of the input."},)
    num_examples_in_instruction: int = field(default=1, metadata={
        "help": "Number of positive/negative examples to add in instructions"},)
    train_test_split_mode: str = field(default="standard_train_test_dev",metadata={
        "help": "Split train and test tasks. default is all tasks are included in train and test. "
        "Other options are random_tasks_zero_shot (no overlap in train and test tasks), "
        "random_categories_zero_shot (no overlap in both tasks and categories"},)
    hnet_add_instruction_to_mainlm: bool = field(default=True, metadata={
        "help": "Whether to add the instruction in front of the input for HNET. By default the instruction is fed to the HNET only"},)  
    # ---------------------- meta parameters -------------------
    num_tasks_per_iter:int = field(default=4, metadata={
        "help": "Number of tasks per minibatch for updates"})
    task_batch_size_per_iter:int = field(default=6, metadata={
        "help": "Number of instances per task in minibatch. This is split into support and "
                "target set. Thus the number needs to be even."})
    task_sampling_mode: Optional[str] = field(default="uniform", metadata={
        "help": "How to sample tasks for n_way k_shot batch. Options: uniform, proportionate"})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def check_training_args(training_args):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            "--overwrite_output_dir to overcome."
        )
    return

def check_model_args(model_args):
    return

def check_data_args(data_args):
    # if data_args.eval_data_file is None and training_args.do_eval:
    #     raise ValueError(
    #         "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #         "or remove the --do_eval argument."
    #     )
    # if data_args.source_prefix is None and model_args.model_name_or_path in [
    #     "t5-small",
    #     "t5-base",
    #     "t5-large",
    #     "t5-3b",
    #     "t5-11b",
    # ]:
    #     logger.warning(
    #         "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
    #         "`--source_prefix 'summarize: ' `"
    #     )
    return

def check_arguments(model_args, data_args, training_args):
    return
    # check_data_args(data_args)
    # check_model_args(model_args)
    # check_training_args(training_args)

