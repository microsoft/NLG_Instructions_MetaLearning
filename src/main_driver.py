"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
"""

import datasets
import json
import logging
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import os
import sys
import pdb
from filelock import FileLock
import copy

from rouge_metric import PyRouge

import transformers
from transformers import EarlyStoppingCallback
from transformers import HfArgumentParser
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from transformers.file_utils import is_offline_mode
from transformers.utils import check_min_version

from learn2learn.algorithms import MAML, MetaSGD

from .arguments import (ModelArguments, DataTrainingArguments, check_arguments,
                       DerivedSeq2SeqTrainingArguments)
from .data_processing import AbstractNIDataHandler
from .derived_trainer import DerivedTrainerClass, TrainerClassMAML, TrainerClassHNET, TrainerClassHNETMAML
from .models_pt import BaseS2SModel, BaseCLMModel
from .models_hnet import S2SHNETModel
from .models_meta import MetaSGD as DerivedMetaSGD


import warnings 
warnings.filterwarnings("ignore")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")
logger = logging.getLogger(__name__)
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first "
            "to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class MainDriver:
    def __init__(self, data_handler_class):
        # parse arguments and check
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                                   DerivedSeq2SeqTrainingArguments))

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            self.model_args, self.data_args, self.training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]))
        else:
            self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses()
            self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses()
        check_arguments(self.model_args, self.data_args, self.training_args)
        self.data_handler_class = data_handler_class
        self.setup_logging()
        self.set_seed()
        # TODO: for now set this to 1, later convert for distributed training
        self.device_count = 1

    def _find_checkpoint(self):
        """Loads last checkpoint according to training arguments."""
        self.last_checkpoint = None
        if (os.path.isdir(self.training_args.output_dir) and self.training_args.do_train and
                not self.training_args.overwrite_output_dir):
            self.last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if self.last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is "
                    "not empty. Use --overwrite_output_dir to overcome."
                )
            elif self.last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                self.logger.info(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. "
                    "To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    def _init_model(self):
        """Create the model if does not exist and selects trainer."""
        if hasattr(self, "model") and self.model is not None:
            return
        self.set_seed()
        # ------------------ create model ------------------
        if ("bart" in self.model_args.model_name_or_path or "allenai/led" in self.model_args.model_name_or_path):
            if self.training_args.train_loop_opt_type in ["standard", "maml"]:
                self.model = BaseS2SModel(self.model_args, self.training_args, self.data_args, self.logger)
            
                if self.training_args.train_loop_opt_type in ["maml"]:
                    # self.model = MAMLS2SModel(self.model, lr=self.training_args.inner_loop_learning_rate,  first_order= not(self.training_args.use_second_order_gradients), allow_unused=False, allow_nograd=False)
                    if self.training_args.use_meta_sgd:
                        self.model = DerivedMetaSGD(self.model, lr=self.training_args.inner_loop_learning_rate,  first_order= not(self.training_args.use_second_order_gradients))
                    else:
                        self.model = MAML(self.model, lr=self.training_args.inner_loop_learning_rate,  first_order= not(self.training_args.use_second_order_gradients), allow_unused=False, allow_nograd=False)
            if self.training_args.train_loop_opt_type in ["hnet", "hnet_maml"]:
                self.model = S2SHNETModel(self.model_args, self.training_args, self.data_args, self.logger)
                
                if self.training_args.train_loop_opt_type in ["hnet_maml"]:
                    if self.training_args.use_meta_sgd:
                        self.model = DerivedMetaSGD(self.model, lr=self.training_args.inner_loop_learning_rate,  first_order= not(self.training_args.use_second_order_gradients))
                    else:
                        self.model = MAML(self.model, lr=self.training_args.inner_loop_learning_rate,  first_order= not(self.training_args.use_second_order_gradients), allow_unused=True, allow_nograd=False)

        if "gpt2" in self.model_args.model_name_or_path:
            self.model = BaseCLMModel(self.model_args, self.training_args, self.data_args, self.logger)
        
        self.print_model_dict(self.model)
        self.tokenizer = self.model.tokenizer
        # ------------------ select trainer class ------------------
        if self.training_args.train_loop_opt_type == "standard":
            self.TrainerClass = DerivedTrainerClass

        if self.training_args.train_loop_opt_type == "maml":
            self.TrainerClass = TrainerClassMAML
        
        if self.training_args.train_loop_opt_type == "hnet":
            self.TrainerClass = TrainerClassHNET

        if self.training_args.train_loop_opt_type == "hnet_maml":
            self.TrainerClass = TrainerClassHNETMAML

    def _init_data_handlers(self):
        """Create the data handler if does not exist."""
        if (not hasattr(self, "data_handler")) or self.data_handler is None:
            if not issubclass(self.data_handler_class, AbstractNIDataHandler):
                raise ValueError("DataHander class should inherit from AbstractNIDataHandler")
            self.data_handler = self.data_handler_class(
                self.model.config, self.model_args, self.training_args, self.data_args, self.logger,
                self.tokenizer, self.model)

    def _init_metrics(self):
        # ------------------ Create Metrics ------------------
        self.metric = datasets.load_metric("rouge")
        # (1) ROUGE-2, ROUGE-L, ROUGE-W, and ROUGE-S worked well in single document
        #   summarization tasks,
        # (2) ROUGE-1, ROUGE-L, ROUGE-W, ROUGE-SU4, and ROUGE-SU9 performed great in evaluating
        #   very short summaries (or headline-like summaries)
        self.rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=False, rouge_w_weight=1.2,
                             rouge_s=False, rouge_su=False, skip_gap=4, multi_ref_mode=self.training_args.multi_ref_mode)
        # PyRouge(rouge_n=(2), rouge_l=True, multi_ref_mode=self.training_args.multi_ref_mode)
        #  ------------------ reduce eval step frequency according to ga steps --------------------
        if self.training_args.gradient_accumulation_steps > 1:
            if self.training_args.evaluation_strategy == "steps":
                self.training_args.eval_steps = (
                    self.training_args.eval_steps / self.training_args.gradient_accumulation_steps)
            if self.training_args.logging_strategy == "steps":
                self.training_args.logging_steps = (
                    self.training_args.logging_steps / self.training_args.gradient_accumulation_steps)
        # list_of_metrics_to_log = ['rouge-2-f', 'rouge-l-f', 'rouge-w-1.2-f',
        # 'rouge-s4-f', 'rouge-su4-f', 'gen_len_pred_avg', 'gen_len_pred_max',
        # 'gen_len_ref_max', 'gen_len_ref_avg', 'diff_pred_ref']
        self.list_of_metrics_to_log = ["rouge-2-f", "rouge-l-f", "diff_pred_ref"]
        self.best_metrics = {k:-1 for k in ["rouge-2-f", "rouge-l-f"]}
        self.best_metrics_overall = {k:-1 for k in ["rouge-2-f", "rouge-l-f"]}

    def save_predictions(self, filename_prefix="predictions_"):
        self._init_model()
        self._init_data_handlers()
        trainer = self.TrainerClass(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_handler.data_collator,
            data_collator_eval=self.data_handler.data_collator_eval,
            compute_metrics=lambda _: {},  # No need to calculate metrics for saving predictions
            callbacks = [])

        output = trainer.predict(self.data_handler.test_dataset)
        _, _, decoded_preds_labels_by_tasks, _ = self.data_handler.test_dataset.decode_labels_and_predictions(
                output.predictions, output.label_ids)

        predictions_dirname = os.path.join(self.training_args.output_dir, "predictions")
        if not os.path.exists(predictions_dirname):
            os.mkdir(predictions_dirname)
        for task, preds_labels in decoded_preds_labels_by_tasks.items():
            task_predictions = [
                {"index": index, "reference": label, "prediction": prediction}
                for index, (label, prediction) in enumerate(zip(preds_labels["labels"], preds_labels["preds"]))
            ]
            predictions_filename = os.path.join(predictions_dirname, f"{filename_prefix}{task}.json")
            with open(predictions_filename, 'w') as pred_file:
                json.dump(task_predictions, pred_file)

    def train(self):
        self._init_model()
        self._find_checkpoint()
        self._init_data_handlers()
        self._init_metrics()
        # ------------------ Initialize our Trainer ------------------
        callback = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.05)
        trainer = self.TrainerClass(
            model=self.model,
            args=self.training_args,
            train_dataset=self.data_handler.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.data_handler.test_dataset if self.training_args.do_eval else None,
            predict_dataset=self.data_handler.eval_dataset if self.training_args.do_eval else None,
            predict_dataset_val=self.data_handler.eval_dataset_val if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_handler.data_collator,
            data_collator_eval=self.data_handler.data_collator_eval,
            compute_metrics=self.compute_metrics if self.training_args.predict_with_generate else None,
            # callbacks = [callback],
            batch_sampler_train = self.data_handler.batch_sampler_train,
            batch_sampler_eval = self.data_handler.batch_sampler_eval)
        # ------------- Training -------------
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint == "True":
                self.training_args.resume_from_checkpoint = True
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            _ = trainer.train(resume_from_checkpoint=checkpoint,)
            trainer.save_model()  # Saves the tokenizer too for easy upload

    def set_seed(self):
        set_seed(self.training_args.seed)

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.training_args.local_rank in [-1, 0] else logging.WARN,
        )
        log_level = self.training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        self.logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.training_args.local_rank,
            self.training_args.device,
            self.training_args.n_gpu,
            bool(self.training_args.local_rank != -1),
            self.training_args.fp16,
        )
        self.logger.info("Training/evaluation parameters %s", self.training_args)

    def flatten_dict(self, rouge_scores):
            flattened_metrics = {}
            for key, val in rouge_scores.items():
                if type(val) is not dict:
                    flattened_metrics[key] = round(val, 4)
                else:
                    for k, v in rouge_scores[key].items():
                        flat_key = "%s-%s"%(key, k)
                        flattened_metrics[flat_key] = round(v, 4)
            return flattened_metrics

    def filter_sequences_by_len_diff_from_ref_task_level(self, decoded_preds_labels):
        decoded_preds_labels_by_tasks = copy.deepcopy(decoded_preds_labels)
        for task, preds_labels in decoded_preds_labels_by_tasks.items():
            for index in range(len(decoded_preds_labels_by_tasks[task]["labels"])):
                # for references iterate within each instance as there can be multiple references 
                avg_len_of_reference = np.mean([len(ref.split()) for ref in decoded_preds_labels_by_tasks[task]["labels"][index]])
                len_of_prediction = len(decoded_preds_labels_by_tasks[task]["preds"][index].split())
                diff_pred_ref = np.abs(avg_len_of_reference - len_of_prediction) / avg_len_of_reference
                if diff_pred_ref > self.training_args.pred_ref_diff_tolerance:
                    decoded_preds_labels_by_tasks[task]["preds"][index] = "***NA***"
        return decoded_preds_labels_by_tasks

    def filter_sequences_by_len_diff_from_ref_all(self, decoded_preds, decoded_labels):
        # decoded_preds_labels_by_tasks = copy.deepcopy(decoded_preds_labels)
        decoded_preds_copy = copy.deepcopy(decoded_preds)

        for index in range(len(decoded_preds_copy)):  
            avg_len_of_reference = np.mean([len(ref.split()) for ref in decoded_labels[index]])
            len_of_prediction = len(decoded_preds_copy.split())
            diff_pred_ref = np.abs(avg_len_of_reference - len_of_prediction) / avg_len_of_reference
            if diff_pred_ref > self.training_args.pred_ref_diff_tolerance:
                decoded_preds_copy[index] = "***NA***"
        return decoded_preds_copy

    def compute_overall_metrics(self, decoded_preds, decoded_labels, eval_preds):
        preds, labels = eval_preds
        rouge_scores_all = self.flatten_dict(self.rouge.evaluate(decoded_preds, decoded_labels))
        rouge_scores = {k:v for k, v in rouge_scores_all.items() if k in self.list_of_metrics_to_log}
        # ------------ compute prediction lenghts -------------
        # rouge_scores = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}
        # prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        # reference_lens = [np.count_nonzero(label != -100) for label in labels]
        # rouge_scores["gen_len_pred_avg"] = np.mean(prediction_lens)
        # rouge_scores["gen_len_ref_avg"] = np.mean(reference_lens)
        # rouge_scores["diff_pred_ref"] = np.abs(
        #     rouge_scores["gen_len_pred_avg"] - rouge_scores["gen_len_ref_avg"]) / rouge_scores["gen_len_ref_avg"]
        return rouge_scores

    def compute_task_level_metrics(self, decoded_preds_labels_by_tasks):
        rouge_scores = {}
        rouge_scores_tasks_dict = {}
        rouge_scores_overall = {k:0.0 for k in self.list_of_metrics_to_log}
        rouge_avgs = {k:0.0 for k in self.list_of_metrics_to_log}
        num_tasks = len(decoded_preds_labels_by_tasks.keys())
        total_items = 0.0
        for task, preds_labels in decoded_preds_labels_by_tasks.items():
            rouge_scores_tasks = self.flatten_dict(self.rouge.evaluate(preds_labels["preds"], preds_labels["labels"]))
            rouge_scores_tasks = {k: v for k, v in rouge_scores_tasks.items() if k in self.list_of_metrics_to_log}
            # # ------------- keep track of overall metrics ----------
            # num_items_in_task = len(preds_labels["labels"])
            # rouge_scores_overall = {k: rouge_scores_overall[k] + v * num_items_in_task for k, v in rouge_scores_tasks.items()}
            # total_items += num_items_in_task
            # ------------ compute avg metrics across tasks --------------
            rouge_avgs = {k: rouge_avgs[k] + v/num_tasks for k, v in rouge_scores_tasks.items()}
            # ------------- set task specific metrics ---------------
            rouge_scores_tasks_dict[task] = {k: v for k, v in rouge_scores_tasks.items()}
            # ---------- update the keys with the task name for logging ----------
            rouge_scores_tasks = {task + '_' + k: v for k, v in rouge_scores_tasks.items()}
            rouge_scores = dict(rouge_scores, **rouge_scores_tasks)
        # # ----------- normalize the rouge_overall for total items ---------
        # rouge_scores_overall = {k : v / total_items for k, v in rouge_scores_overall.items()}
        # ----------- combine the dicts before returning ------------
        rouge_scores = dict(rouge_scores, **rouge_avgs)
        rouge_scores_tasks_dict["avg_across_tasks"] = rouge_avgs
        # rouge_scores["rouge_scores_overall"] = rouge_scores_overall
        return rouge_scores, rouge_scores_tasks_dict, rouge_avgs

    def compute_metrics(self, eval_preds, dataloader=None):
        preds, labels = eval_preds
        # pdb.set_trace()
        # dataset = dataloader.dataset
        decoded_preds, decoded_labels, decoded_preds_labels_by_tasks, \
            decoded_preds_labels_indexes_by_tasks = dataloader.dataset.decode_labels_and_predictions(preds, labels)
        if self.training_args.log_task_level_metrics:
            if self.training_args.filter_sequences_by_len_diff_from_ref:
                decoded_preds_labels_by_tasks_filtered = self.filter_sequences_by_len_diff_from_ref_task_level(decoded_preds_labels_by_tasks)
                rouge_scores, rouge_scores_tasks_dict, rouge_avgs = self.compute_task_level_metrics(decoded_preds_labels_by_tasks_filtered)
            else:
                rouge_scores, rouge_scores_tasks_dict, rouge_avgs = self.compute_task_level_metrics(decoded_preds_labels_by_tasks)
        else:
             if self.training_args.filter_sequences_by_len_diff_from_ref:
                decoded_preds_copy = self.filter_sequences_by_len_diff_from_ref_all(decoded_preds, decoded_labels)
        # ------------- also compute overall metrics and add to the dict ----------------
        rouge_scores_overall = self.compute_overall_metrics(decoded_preds, decoded_labels, eval_preds)
        rouge_scores_tasks_dict["rouge_scores_overall"] = rouge_scores_overall
        rouge_scores["rouge_scores_overall"] = rouge_scores_overall
        # rouge_scores = dict(rouge_scores_overall, **rouge_scores)
        # ------------------ get target task sampling rates during training ----------------
        target_tast_sampling_rate =  self.data_handler.get_target_test_sampling_rate()
        if target_tast_sampling_rate is not None:
            for task, value in target_tast_sampling_rate.items():
                rouge_scores["sampling_rate_%s"%task] = value
        # ------------------ Extract a few results from ROUGE ------------------
        self.data_handler.print_sample_predictions(decoded_preds_labels_by_tasks)
        # ------- update the best metrics -------
        for k, v in self.best_metrics.items():
            if rouge_avgs[k] >= v:
                self.best_metrics[k] = rouge_avgs[k]
                if "rouge-l" in k:
                    self.save_best_predictions(filename_prefix = "predictions_best_across_tasks_", decoded_preds_labels_by_tasks=decoded_preds_labels_by_tasks, dataset=dataloader.dataset)
        for k, v in self.best_metrics_overall.items():
            if rouge_scores_overall[k] >= v:
                self.best_metrics_overall[k] = rouge_scores_overall[k]
                if "rouge-l" in k:
                    self.save_best_predictions(filename_prefix = "predictions_best_overall_", decoded_preds_labels_by_tasks=decoded_preds_labels_by_tasks, dataset=dataloader.dataset)

        rouge_scores_tasks_dict["best_metrics"] = self.best_metrics
        rouge_scores_tasks_dict["best_metrics_overall"] = self.best_metrics_overall
        rouge_scores = dict(rouge_avgs, **rouge_scores)
        rouge_scores["best_metrics"] = self.best_metrics
        rouge_scores["best_metrics_overall"] = self.best_metrics_overall
        # ------- pretty print the metrics -------
        print(f"==========================================================")
        print_str = "task" + "\t" + "\t".join(self.list_of_metrics_to_log)
        for task, scores in rouge_scores_tasks_dict.items():
            print_str += "\n%s\t"%task     
            for k, v in scores.items():
                print_str += "%s\t"%v
        print(print_str)
        print(f"==========================================================")
        # ------ flatten dict before returning to log correctly ---------
        rouge_scores = self.flatten_dict(rouge_scores)
        return rouge_scores

    def save_best_predictions(self, filename_prefix="predictions_", decoded_preds_labels_by_tasks=None, dataset=None):
        predictions_dirname = os.path.join(self.training_args.output_dir, "predictions")
        if not os.path.exists(predictions_dirname):
            os.mkdir(predictions_dirname)
        for task, preds_labels_inputs in decoded_preds_labels_by_tasks.items():
            task_predictions = [
                {"index": index, "input": input, "reference": label, "prediction": prediction}
                for index, (input, label, prediction) in enumerate(zip(preds_labels_inputs["inputs"], preds_labels_inputs["labels"], preds_labels_inputs["preds"]))]
            task_predictions_with_instructions = [dataset.instance_full_task_description[task]] + task_predictions
            predictions_filename = os.path.join(predictions_dirname, f"{filename_prefix}{task}.json")
            with open(predictions_filename, 'w') as pred_file:
                json.dump(task_predictions_with_instructions, pred_file)

    def print_model_dict(self, model):
        print(f"---------------------------------------------------------")
        print(f"Defined model with state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(f"---------------------------------------------------------")

    def run(self):
        self.train()