
import json
import os
import nltk
import numpy
import random
import time
import torch
import pdb 
import copy

from src import data_processing
from torch.utils.data import Dataset

from .task_list import (TRAIN_TASKS_LIST_SUMM, TEST_TASKS_LIST_SUMM, TRAIN_TASKS_LIST_SUMM_TITLE,
                        TEST_TASKS_LIST_SUMM_TITLE, TASK_LIST_GENERATION, TASK_LIST_GENERATION_V2_5, EXCLUDE_TASKS_LIST_SUMM_TITLE)


# TODO split this class with data handling functions on one side and train/test logic on the other
class DataHandlerNaturalInstructions(data_processing.AbstractNIDataHandler):

    def __init__(self, model_config, model_args, training_args, data_args, logger, tokenizer, model,
                 return_tensors="pt"):
        super().__init__(model_config, model_args, training_args, data_args, logger,
                         tokenizer, model, return_tensors)
        self.allowed_task_languages = set(["English"])
        random.seed(0)
        # -------------split tasks into train and test---------------
        self.train_test_task_split = 0.8
        self.get_list_of_tasks()
        self.split_tasks_to_train_test(split_type=self.data_args.train_test_split_mode)
        # ------------- create the train, dev, test datasets ---------------
        self.create_datasets()
        # ------------- Create a Batch Sampler for Standard and Meta-Learning loops -------------
        self.create_batch_samplers_and_collators()

    def _input_metadata_filename(self):
        return os.path.join(self.data_args.dataset_folder, "tasks", "tasks_meta_data.json")

    @classmethod
    def read_task_metadata(cls, input_filename, allowed_task_names=None, allowed_task_languages=None):
        """
            Reads the tasks from the @input_filename.
            Filter tasks by language and by Blocklist.
            Args:
                input_filename (str): path to .json file with task metadata
                allowed_task_names (set): Names of tasks allowed. If None or empty, all
                    languages are included.
                allowed_task_languages (set): Task languages allowed. If None or empty, all
                    languages are included.

            Returns:
                A dictionary where keys are task names and values are task metadata.
        """
        with open(input_filename, encoding="utf-8") as json_file:
            tasks = json.load(json_file)
        task_count = 0
        # ---------- split the categories into hierarchi and do basic cleanup ----------
        for task_name, task_value in tasks.items():
            category_dict = {}
            # -------- create structure for storing categories hierarchically ------------
            # NOTE: This is not required in the release version as categories have been cleaned up
            # for category in tasks[task_name]["Categories"]:
            #     cat_levels = [cat.strip() for cat in category.split("->")]
            #     for index in range(len(cat_levels)):
            #         if "level_%d"%index in category_dict:
            #             category_dict["level_%d"%index].append(cat_levels[index])
            #         else:
            #             category_dict["level_%d"%index] = [cat_levels[index]]
            # tasks[task_name]["Hirarchical_Categories"] = category_dict

            if "Domains" not in tasks[task_name]:
                tasks[task_name]["Domains"] = "None"
            else:
                tasks[task_name]["Domains"] = tasks[task_name]["Domains"]

            tasks[task_name]["index"] = task_count
            tasks[task_name]["num_instances_per_task"] = 1.0  # All instances in task
            task_count += 1
        # ------------- filter tasks by allowlist and language ------------------
        list_of_tasks = list(tasks.keys())
        for task_name in list_of_tasks:
            if allowed_task_names is not None and len(allowed_task_names) > 0:
                if task_name not in allowed_task_names:
                    tasks.pop(task_name)
                    continue
            task_languages = set(
                tasks[task_name]["Input_language"] +
                tasks[task_name]["Output_language"] +
                tasks[task_name]["Instruction_language"])
            # ---------- compute the difference of the sets and it should be exact matches ----------
            if task_languages is not None and len(task_languages) > 1:
                #  ------- compute diff of the two sets -----------
                if len(allowed_task_languages ^ task_languages) == 0:
                    continue
                else:
                    tasks.pop(task_name)
        # ------------- filter by block list of tasks -----------
        return tasks

    def get_list_of_tasks(self):
        """
            Read the tasks from the tasks_meta_data.json, and populate a dict for general processing.
            Filter tasks by language
            Filter tasks by Blocklist
        """
        self.tasks = self.read_task_metadata(
            self._input_metadata_filename(), allowed_task_languages=self.allowed_task_languages)

    def split_tasks_to_train_test(self, split_type="random_zero_shot"):
        """Split tasks into different buckets for train and test.

        Train tasks can have all, subeset of tasks, and subset of categories, from the train split of the dataset.
        To be able to compare these three settings on the same dev/test set, we consider the smallest task set,
        which comprises of all tasks filtered by training categories.
        """
        # ------------ first randomly divide tasks into train and test --------------
        all_task_name_list = [name for name in self.tasks.keys()]
        random.seed(0)
        random.shuffle(all_task_name_list)
        num_train_tasks = int(len(all_task_name_list) * self.train_test_task_split)
        # --------------------- Create Train Tasks --------------------
        if self.data_args.use_train_tasks_list == "TRAIN_TASKS_LIST_SUMM":
            self.train_tasks_names = TRAIN_TASKS_LIST_SUMM
        elif self.data_args.use_train_tasks_list == "TRAIN_TASKS_LIST_SUMM_TITLE":
            self.train_tasks_names = TRAIN_TASKS_LIST_SUMM_TITLE
        elif self.data_args.use_train_tasks_list == "TASK_LIST_GENERATION":
            self.train_tasks_names = TASK_LIST_GENERATION
        elif self.data_args.use_train_tasks_list == "TASK_LIST_GENERATION_V2_5":
            self.train_tasks_names = TASK_LIST_GENERATION_V2_5
        elif self.data_args.use_train_tasks_list == "False":
            self.train_tasks_names = all_task_name_list
        else: # --------- train_task_list provides a comma separated a list of tasks ---------------
            self.train_tasks_names = self.data_args.use_train_tasks_list.split(",")
        # --------------------- Create Eval Tasks --------------------
        if self.data_args.use_eval_tasks_list == "TEST_TASKS_LIST_SUMM_TITLE":
            self.eval_task_names = TEST_TASKS_LIST_SUMM_TITLE
        # --------------------- Create Test Tasks --------------------
        if self.data_args.use_test_tasks_list == "TEST_TASKS_LIST_SUMM":
            self.test_tasks_names = TEST_TASKS_LIST_SUMM
        elif self.data_args.use_test_tasks_list == "TEST_TASKS_LIST_SUMM_TITLE":
            self.test_tasks_names = TEST_TASKS_LIST_SUMM_TITLE
        elif self.data_args.use_test_tasks_list == "TASK_LIST_GENERATION":
            self.test_tasks_names = TASK_LIST_GENERATION
        elif self.data_args.use_test_tasks_list == "False":
            potential_test_list = TASK_LIST_GENERATION_V2_5
            random.shuffle(potential_test_list)
            num_train_tasks = int(len(potential_test_list) * self.train_test_task_split)
            self.test_tasks_names = potential_test_list[num_train_tasks:] # we select the train tasks based on the mode
            # self.test_tasks_names = self.train_tasks_names[num_train_tasks:] # we select the train tasks based on the mode
        else: # --------- test_tasks task list provides a comma separated list of tasks ---------------
            self.test_tasks_names = self.data_args.use_test_tasks_list.split(",")
        # --------------------- Create Exclude Tasks --------------------
        if self.data_args.use_exclude_tasks_list == "TEST_TASKS_LIST_SUMM":
            self.exclude_tasks = TEST_TASKS_LIST_SUMM
        elif self.data_args.use_exclude_tasks_list == "EXCLUDE_TASKS_LIST_SUMM_TITLE":
            self.exclude_tasks = EXCLUDE_TASKS_LIST_SUMM_TITLE
        elif self.data_args.use_exclude_tasks_list == "TASK_LIST_GENERATION":
            self.exclude_tasks = TASK_LIST_GENERATION
        elif self.data_args.use_exclude_tasks_list == "False":
            self.exclude_tasks = []
        else: # --------- task list provides a comma separated list of tasks ---------------
            self.exclude_tasks = self.data_args.use_exclude_tasks_list.split(",")
        # --------------------- Exclude Tasks from Train list --------------------
        """ 
            Exclude from  Train list if task is not also a test_task
            test_tasks themselves can be excluded by controlling the kshot parameter, so even if it is included
            in the training.
            It may not be used if the kshot parameter is set to zero
        """
        self.exclude_tasks = list(set(self.exclude_tasks) - set(self.test_tasks_names))
        self.train_tasks_names = [task for task in self.train_tasks_names if task not in self.exclude_tasks]
        # assert(self.test_tasks_names[0] in self.train_tasks_names)
        # if len(self.exclude_tasks) > 0: # not in self.test_tasks_names:
        #     assert(self.exclude_tasks[0] not in self.train_tasks_names)
        print("---------------------------------------------------------------------------------")
        print("Train Tasks: %d: ", self.train_tasks_names)
        print("---------------------------------------------------------------------------------")
        print("Test Tasks", self.test_tasks_names)
        print("---------------------------------------------------------------------------------")
        print("Eval Tasks", self.eval_task_names)
        print("---------------------------------------------------------------------------------")
        print("Excluded Tasks", self.exclude_tasks)
        print("---------------------------------------------------------------------------------")
        # ---------- After split, create the dictionaries for train and test tasks --------------------
        self.train_tasks = {}
        self.test_tasks = {}
        self.eval_tasks = {}

        # fp_train = open(os.path.join(self.training_args.output_dir, "train_tasks.tsv"), "w")
        # fp_valid = open(os.path.join(self.training_args.output_dir, "valid_tasks.tsv"), "w")
        # fp_eval = open(os.path.join(self.training_args.output_dir, "eval_tasks.tsv"), "w")
        for task_name, values in self.tasks.items():
            # The following will be used in k-shot training. The train tasks will contain all the data from training.
            # While tasks in test, may contain zero to k instances of the task. We use this indicator when creating
            # the dataset for including the number of instances for each task.
            if task_name in self.train_tasks_names:
                self.train_tasks[task_name] = {}
                self.train_tasks[task_name]["index"] = values["index"]
                self.train_tasks[task_name]["Summary"] = values["Summary"]
                self.train_tasks[task_name]["Categories"] = values["Categories"]
                # self.train_tasks[task_name]["Hirarchical_Categories"] = values["Hirarchical_Categories"]
                if (task_name in self.test_tasks_names) or (task_name in self.eval_task_names):
                    self.train_tasks[task_name]["num_instances_per_task"] = self.data_args.num_kshot_train_instances_per_task
                    self.train_tasks[task_name]["in_split"] = "test"
                else:
                    self.train_tasks[task_name]["num_instances_per_task"] = self.data_args.num_train_instances_per_task
                    self.train_tasks[task_name]["in_split"] = "train"
                # fp_train.write("%s\t%s\n"%(task_name,  values["Summary"]))

            if task_name in self.test_tasks_names:
                self.test_tasks[task_name] = {}
                self.test_tasks[task_name]["Summary"] = values["Summary"]
                self.test_tasks[task_name]["Categories"] = values["Categories"]
                self.test_tasks[task_name]["index"] = values["index"]
                self.test_tasks[task_name]["in_split"] = "test"
                self.test_tasks[task_name]["num_instances_per_task"] = self.data_args.num_test_instances_per_task
                # fp_valid.write("%s\t%s\n"%(task_name,  values["Summary"]))

            if task_name in self.eval_task_names:
                self.eval_tasks[task_name] = {}
                self.eval_tasks[task_name]["Summary"] = values["Summary"]
                self.eval_tasks[task_name]["Categories"] = values["Categories"]
                self.eval_tasks[task_name]["index"] = values["index"]
                self.eval_tasks[task_name]["in_split"] = "test"
                self.eval_tasks[task_name]["num_instances_per_task"] = self.data_args.num_eval_instances_per_task
                # fp_eval.write("%s\t%s\n"%(task_name,  values["Summary"]))
        
        # fp_train.close()
        # fp_valid.close()
        # fp_eval.close()

    def create_datasets(self):
        """
            NOTE: We use the Test dataset for validations and Dev data split for evaluation. Need change this later. 
        """
        print("Loading datasets for Train")
        self.train_dataset = DatasetNaturalInstructions(
            self.model_config,
            os.path.join(self.dataset_folder, "tasks", "train"),
            self.train_tasks,
            self.model_args, self.training_args, self.data_args, self.logger, self.tokenizer, "train")
        print("-------------------------------------------------------------------------------")
        # test tasks should be the different from train tasks
        print("Loading datasets for Test")
        self.test_dataset = DatasetNaturalInstructions(
                self.model_config,
                os.path.join(self.dataset_folder, "tasks", "test"),
                self.test_tasks,
                self.model_args, self.training_args, self.data_args, self.logger, self.tokenizer, "eval")
        # dev tasks should be the same as train tasks ideally, for now just set to test task
        print("Loading datasets for Eval")
        if self.data_args.use_train_dataset_for_eval:
            self.eval_dataset = DatasetNaturalInstructions(
                self.model_config,
                os.path.join(self.dataset_folder, "tasks", "train"),
                self.eval_tasks,
                self.model_args, self.training_args, self.data_args, self.logger, self.tokenizer, "eval")
        else:
            self.eval_dataset = DatasetNaturalInstructions(
                self.model_config,
                os.path.join(self.dataset_folder, "tasks", "dev"),
                self.eval_tasks,
                self.model_args, self.training_args, self.data_args, self.logger, self.tokenizer, "eval")
        # this is the dataset for validation tasks
        print("Loading datasets for Eval (Validation tasks")
        self.eval_dataset_val = DatasetNaturalInstructions(
                self.model_config,
                os.path.join(self.dataset_folder, "tasks", "dev"),
                self.test_tasks,
                self.model_args, self.training_args, self.data_args, self.logger, self.tokenizer, "eval")

        print("-------------------------------------------------------------------------------")
        self.train_dataset.select_items(self.data_args.max_train_samples)
        self.test_dataset.select_items(self.data_args.max_eval_samples)
        self.eval_dataset.select_items(self.data_args.max_eval_samples)
        print("Loaded Train [%d], Test [%d], Eval [%d], Eval(Val Tasks) [%d] instances in each"%(len(self.train_dataset),
                                                                        len(self.test_dataset),
                                                                        len(self.eval_dataset), len(self.eval_dataset_val)))
        print("-------------------------------------------------------------------------------")

    def print_sample_predictions(self, decoded_preds_labels_by_tasks):
        if self.training_args.show_example_predictions <= 0:
            return
        for task, preds_labels in decoded_preds_labels_by_tasks.items():
            num_to_print = min(self.training_args.show_example_predictions, len(preds_labels['preds']))
            print(f"==========================================================")
            print("Task:%s, Summary: %s"%(task, self.tasks[task]["Summary"]))
            for index in range(num_to_print):
                print(f"---------------------------------------------------------")
                print("[Prediction]", preds_labels["preds"][index])
                print("[Reference]", preds_labels["labels"][index])
            print(f"==========================================================")

class DatasetNaturalInstructions(Dataset):

    TRAIN_MODE = "train"
    EVAL_MODE = "eval"

    def __init__(self, model_config, dataset_folder, tasks, model_args, training_args, data_args, logger,
                 tokenizer, dataset_mode, input_prefix="[Input]:", output_prefix="[Output]:"):
        self.model_config = model_config
        self.dataset_folder = dataset_folder
        self.tasks = tasks
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataset_mode = dataset_mode
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        # self.instruction_structure =['Definition','Things to Avoid','Emphasis & Caution',
        #                              'Negative Examples Full Explanations', 'Positive Examples Full Explanations']
        self.load_datasets(self.tasks)
        # NOTE: DO NOT shuffle the dataset as instance indexes gets messed up.
        # self.shuffle_dataset_instances()

    def select_items(self, num_items):
        # select all items if the num_items is set to -1
        if num_items is None:
            num_items = len(self.instances_task_name)
        else:
            num_items = min(num_items, len(self.instances_task_name))
        self.total_instances = num_items

    def __len__(self):
        return self.total_instances

    def __getitem__(self, index):
        """
        Returns:
            - a dict with key = task_id
                - there are task_batch_size_per_iter tasks with task_ids as the key
            - list_of features[task_id] is a list of dicts
            - list_of features[0][task_id][index] is a dict
                - with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                - has list of tensors of length task_batch_size_per_iter
        """
        if type(index) is not list:
            output = {
                "task_name": self.instances_task_name[index],
                "source": self.instances_source[index],
                "category": self.instances_category[index],
                "positive_example": self.instance_positive_examples[self.instances_task_name[index]],
                "negative_example": self.instance_negative_examples[self.instances_task_name[index]],
                "target": self.instances_target[index],
                "instruction": self.instance_instruction[self.instances_task_name[index]],
                "task_summary": self.tasks[self.instances_task_name[index]]["Summary"],
                "task_index": self.instances_task_indexes[index]
            }
            return self.preprocess_function(output)
        else:
            if (self.training_args.train_loop_opt_type in ["maml", "hnet_maml"]) or (self.training_args.train_loop_opt_type in ["hnet"]and self.training_args.run_hnet_in_batch_mode_per_task==True) or (self.training_args.train_loop_opt_type in ["standard"] and self.training_args.run_hnet_in_batch_mode_per_task==True):
                # for fewshotbatchsampler, it returns a list of indices which we can set equal to the batch size
                # We divide the indexes and create a dict indexesx by the task ID, with each item containing
                # k_shot instances of each task
                model_inputs_by_task = {}
                for idx in index:
                    output = {
                        "task_name":self.instances_task_name[idx],
                        "source":self.instances_source[idx],
                        "category":self.instances_category[idx],
                        "target":self.instances_target[idx],
                        "positive_example": self.instance_positive_examples[self.instances_task_name[idx]],
                        "negative_example": self.instance_negative_examples[self.instances_task_name[idx]],
                        "instruction":self.instance_instruction[self.instances_task_name[idx]],
                        "task_summary": self.tasks[self.instances_task_name[idx]]["Summary"],
                        "task_index": self.instances_task_indexes[idx]
                    }
                    model_input = self.preprocess_function(output)
                    task_index_str = str(output["task_index"])
                    if task_index_str not in model_inputs_by_task:
                        model_inputs_by_task[task_index_str] = [model_input]
                    else:
                        model_inputs_by_task[task_index_str].append(model_input)
                return model_inputs_by_task

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
            Shift input ids one token to the right.
        """
        decoder_input_ids = torch.LongTensor(input_ids)
        shifted_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
        # shifted_input_ids[:, 1:] = decoder_input_ids[:, :-1].clone()
        shifted_input_ids[1:] = decoder_input_ids[:-1].clone()
        shifted_input_ids[0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids.tolist()

    def format_instruction(self, example):
        # ------------------ format the input string -------------------
        """
            Instruction contains task_name(default), and optionally the category, task summary, detailed instructions,
            positive examples and negative examples.
            These are all added in the function encode_instruction_and_instances.
            Eac component are seprated by newlines, so this is not required here.
        """
        # --------- add instruction -------------
        instruction = example["instruction"]
        # --------- add positive example -------------
        random.seed(0)
        if self.data_args.add_positive_examples:
            positive_examples = example["positive_example"]
            if self.data_args.num_examples_in_instruction > 0:
                instruction += "[Positive Examples]:\n"
                num_examples = min(self.data_args.num_examples_in_instruction, len(positive_examples))
                random.shuffle(positive_examples)
                for index in range(num_examples):
                    instruction += positive_examples[index]
        # --------- add negative example -------------
        if self.data_args.add_negative_examples:
            negative_examples = example["negative_example"]
            if self.data_args.num_examples_in_instruction > 0:
                instruction += "[Negative Examples]:\n"
                num_examples = min(self.data_args.num_examples_in_instruction, len(negative_examples))
                random.shuffle(negative_examples)
                for index in range(num_examples):
                    instruction += negative_examples[index]
        return instruction

    def tokenize_strings_clm(self, input, label_string, instruction):
        """
            Here we encode the source and target in in the same input IDS then mask the label string to just the input IDs
        """
        padding = "max_length" if self.data_args.pad_to_max_length else False
        model_inputs = {}
        # edited_sents = '{} [BOS]'.format(input) + label_string + ' [EOS]'
        # edited_sents = '{} '.format(input) + " {}".format(label_string) # this does not work
        # edited_sents = '{} '.format(input) + " {} {}".format(label_string, self.tokenizer.eos_token) # works 
        # Special begin and end tokens need to be added to the GPT2 tokenizer as it does not add it by default
        edited_sents = '{} '.format(input) + "{} {} {}".format(self.tokenizer.bos_token, label_string, self.tokenizer.eos_token) 
        # ------------ tokenize the input to calculate the number of tokens adding an extra ]\n ----------
        tokenized_inputs = self.tokenizer("%s\n"%input, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        tokenized_edited_sent = self.tokenizer(edited_sents, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        # ---------- create the label tensor with input masked out -----------
        labels = copy.deepcopy(tokenized_edited_sent['input_ids'])
        sep_idx = len(tokenized_inputs['input_ids'])
        labels[:sep_idx] = [-100] * sep_idx
        # -------------- create model input dict -----------
        model_inputs["input_ids"] = tokenized_edited_sent["input_ids"]
        model_inputs["attention_mask"] = tokenized_edited_sent["attention_mask"]
        model_inputs["labels"] = labels
        # decoder_input_ids is not used in GPT2, but need to add to keep the inputs to the S2S and CLM same
        model_inputs['decoder_input_ids'] = labels

        return model_inputs

    def tokenize_strings_s2s(self, input, label_string, instruction):
        """
            Here we encode the source and target in separate tensors
        """
        # ------------ create model input dict ------------
        padding = "max_length" if self.data_args.pad_to_max_length else False
        model_inputs = {}
        # --------- tokenize input -------------
        # Special begin and end tokens DOES NOT need to be added to the BART tokenizer as it adds it by default
        tokenized_inputs = self.tokenizer(input, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        model_inputs["input_ids"] = tokenized_inputs["input_ids"]
        model_inputs["attention_mask"] = tokenized_inputs["attention_mask"]
        # ------------------ Setup the tokenizer for targets ------------------
        # NOTE for decoder prepending we would need to reduce the length of the target length by decode length and
        # then add the label.
        if label_string is not None:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(label_string, max_length=self.data_args.max_target_length, padding=True, truncation=True)
                # ------------ If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
                # when we want to ignore padding in the loss.------------
                if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                    labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                                           for label in labels["input_ids"]]
                model_inputs["labels"] = labels["input_ids"]
        else:
            """
                # NOTE: Hardcoding it to the decoding start string but should be programmatic. Something like this
                decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
                decoder_input_ids = (torch.ones((input_ids.shape[0], 1),
                                     dtype=torch.long, device=input_ids.device) * decoder_start_token_id)
            """
            model_inputs["labels"] = [2]
        # ----------- specific to longformer ----------
        if "allenai/led" in self.model_args.model_name_or_path:
            # ------------ create 0 global_attention_mask lists ------------
            model_inputs["global_attention_mask"] = [0 for _ in range(self.data_args.max_source_length)]
            # ------------  since above lists are references, the following line changes the 0 index for all samples ------------
            model_inputs["global_attention_mask"][0] = 1

        """
            Mask the label tokens for the instruction part so that it is not used in loss computation
            Instruction tokens would have an extra end token, so we only count till the second last token
            replace all instruction tokens in the labels by -100
        """
        if (self.training_args.prepend_inst_in_enc_or_dec == "decoder" and
                self.training_args.mask_label_for_decoder_inst is True):
            model_inputs["decoder_input_ids"] = self.shift_tokens_right(
                model_inputs["labels"], self.tokenizer.pad_token_id, self.model_config.decoder_start_token_id)
            instruction_tokens = self.tokenizer(
                instruction, max_length=self.data_args.max_target_length, padding=True, truncation=True)["input_ids"]
            model_inputs["labels"][0:len(instruction_tokens)-1] = [-100] * (len(instruction_tokens)-1)

        return model_inputs

    def preprocess_function(self, example):
        """
            examples = {"task_name":self.instances_task_name[index],
                        "category":self.instances_category[index],
                        "source":self.instances_source[index],
                        "target":self.instances_target[index],
                        "instruction":self.instance_instruction[self.instances_task_name[index]]}
        """
        instruction = self.format_instruction(example)

        if self.training_args.train_loop_opt_type in ["standard", "maml"]: # both these use the same input+instruction format
            if self.training_args.prepend_inst_in_enc_or_dec == "encoder":
                input = instruction + example["source"]
                if self.dataset_mode == self.TRAIN_MODE:
                    label_string = example["target"]
                else:
                    # NOTE: Setting this to the prefix string primes the decoder and shows significantly
                    # improved performance and stability.
                    label_string = self.output_prefix
            elif self.training_args.prepend_inst_in_enc_or_dec == "decoder":
                #  ---------- always add the task name in front of the encoder ----------
                input = "[Task]: %s/n"%example["task_name"] + example["source"]
                if self.dataset_mode == self.EVAL_MODE:
                    #  In the eval mode, we prompt the decoder with instruction, and the output_prefix
                    instruction = instruction + self.output_prefix
                    label_string = instruction
                else:
                    #  In the train mode, the decoder has the instruction _ decoder_prefix + label_string
                    label_string = instruction + example["target"]   
        if self.training_args.train_loop_opt_type in ["hnet", "hnet_maml"]: # should create a separate input for hnet
            # TODO: Probably should parameterize what gets fed to the mainlm and hnet separately
            if self.data_args.hnet_add_instruction_to_mainlm:
                input = instruction + example["source"]
            else:
                input = example["source"]
            if self.dataset_mode == self.TRAIN_MODE:
                    label_string = example["target"]
            else:
                label_string = self.output_prefix
        # ------------ tokenize inputa and output ------------
        if self.training_args.train_loop_opt_type in ["standard", "maml"]: # both these use the same input+instruction format
            if ("bart" in self.model_args.model_name_or_path or
                    "allenai/led" in self.model_args.model_name_or_path):
                model_inputs = self.tokenize_strings_s2s(input, label_string, instruction)
            if "gpt2" in self.model_args.model_name_or_path:
                model_inputs = self.tokenize_strings_clm(input, label_string, instruction)
        if self.training_args.train_loop_opt_type in ["hnet", "hnet_maml"]: # should create a separate input for hnet
            #  we pass an empty string for the instruction
            if ("bart" in self.model_args.model_name_or_path or
                    "allenai/led" in self.model_args.model_name_or_path):
                model_inputs = self.tokenize_strings_s2s(input, label_string, "")
            if "gpt2" in self.model_args.model_name_or_path:
                model_inputs = self.tokenize_strings_clm(input, label_string, "")
            # -------------- tokenize the instructions for the hnet ------------ 
            tokenized_inputs_hnet = self.tokenizer(instruction, max_length=self.data_args.max_source_length, padding=True, truncation=True)
            model_inputs['input_ids_hnet'] = tokenized_inputs_hnet['input_ids']
            model_inputs['attention_mask_hnet'] = tokenized_inputs_hnet['attention_mask']

        return model_inputs

    def shuffle_dataset_instances(self):
        """
            We shuffle the flat lists of instances such that if a subset of the data is
            selected for training or validation, it comes from a random representative set of the entire dataset.
        """
        lists = zip(self.instances_source, self.instances_target, self.instances_category,
                    self.instances_task_name, self.instances_source_to_targets)
        lists = [l for l in lists]
        random.seed(0)
        random.shuffle(lists)
        
        self.instances_source, self.instances_target, self.instances_category, self.instances_task_name, \
                self.instances_source_to_targets = zip(*lists)

    def load_datasets(self, tasks):
        """Prepares the datastructures to index instances.

        To enable batch sampler to sample 1) tasks and 2) instances within each task, we create a flat list of
        instances (source and target), and corresponding lists of categories and task names for each task
        Create a list of
            - instances_source: list of strings
            - instances_target: list of strings
            - instances_category: list of category of each source
            - instances_task_name: list of task name for each source
            - instructions: dictionary of instructions keyed by task name
        """
        self.total_instances = 0
        self.instances_source = []
        self.instances_target = []
        self.instances_source_to_targets = []
        self.instances_category = []
        self.instances_task_name = []
        self.instances_task_indexes = []
        self.instance_instruction = {}
        self.instance_full_task_description = {}
        

        self.instance_positive_examples = {}
        self.instance_negative_examples = {}

        self.num_instances_per_task = {}
        self.num_instances_per_category = {}
        self.category_mapping = {}
        self.categories_from_json = {}
        self.num_instances_per_category = {}
        # --------------loop over the dataset names and create instances--------------
        for task_name, task_values in tasks.items(): #task lists keys are the names of the tasks
            self._load_task(task_name, task_values)

    def _load_task(self, task_name, task_values):
        json_file_name = os.path.join(self.dataset_folder, task_name + ".json")
        if os.path.exists(json_file_name):
            if task_values["num_instances_per_task"] <= 0.0:
                return

            generic_instruction, sources, targets, source_to_targets, positive_examples, negative_examples = \
                self.encode_instruction_and_instances(
                    json_file_name=json_file_name, task_name=task_name, task_summary=task_values["Summary"],
                    number_of_examples=self.data_args.num_examples_in_instruction,
                    number_of_instances=task_values["num_instances_per_task"])

            if len(sources) <= 1:
                return

            self.instances_source.extend(sources)
            self.instances_target.extend(targets)
            # For eval mode each source can have multiple targets. So we keep a separate list for that
            self.instances_source_to_targets.extend(source_to_targets)
            # create flat list of categories
            # self.instances_category.extend(
            #     [task_values["Hirarchical_Categories"]["level_0"][0]]*len(sources))
            self.instances_category.extend([task_values["Categories"]]*len(sources))
            # create flat list of task names for each source
            self.instances_task_name.extend([task_name]*len(sources))
            # create flat list of task names for each source
            self.instances_task_indexes.extend([task_values["index"]]*len(sources))
            self.instance_instruction[task_name] = generic_instruction
            self.instance_positive_examples[task_name] = positive_examples
            self.instance_negative_examples[task_name] = negative_examples
            # --------------- count instances ---------------
            self.num_instances_per_task[task_name] = len(sources)
            # if task_values["Hirarchical_Categories"]["level_0"][0] in self.num_instances_per_category:
            #     self.num_instances_per_category[task_values["Hirarchical_Categories"]["level_0"][0]] += len(sources)
            # else:
            #     self.num_instances_per_category[task_values["Hirarchical_Categories"]["level_0"][0]] = len(sources)
            # if task_values["Categories"] in self.num_instances_per_category:
            #     self.num_instances_per_category[task_values["Categories"]] += len(sources)
            # else:
            #     self.num_instances_per_category[task_values["Categories"]] = len(sources)
            self.total_instances += len(sources)
            assert (self.total_instances == len(self.instances_source))
            # assert (self.total_instances == len(self.instances_category))
            assert (self.total_instances == len(self.instances_task_name))
            assert (self.total_instances == len(self.instances_source_to_targets))
            # print("Loaded dataset [%s], category = [%s], num_tasks [%d], Total Instances [%d]"%(
            #     task_name, task_values["Hirarchical_Categories"]["level_0"][0],
            #     len(sources), self.total_instances))
            # # --------------- write data stats to a file ---------------
            # if self.dataset_mode == "train":
            #     fp.write("%s\t%s\t%s\t%s\t%d\n"%(task_name, values["summary"],
            #                                      values["category"], category[0], len(sources)))
        else:
            print("Warning: Loaded dataset [%s], data file missing!!"%task_name)

    def encode_instruction_and_instances(
            self, json_file_name=None, task_name=None, task_summary=None,
            number_of_examples=0, number_of_instances=0):
        """Encode the instructions, sources and targets into some prescribed format from the json file for each task.

        Instructions have the following components: ['Categories', 'Definition', 'Positive Examples',
            'Negative Examples', 'Instances']
        Positive and Negative Examples have: ["input", "output", "explanation"]
        """
        # ----------------- read json file for the task name -----------------
        with open(json_file_name) as json_file:
            json_data_dict = json.load(json_file)
        # ----------------- store the full task description -----------------
        task_keys = ['Summary', 'Categories', 'Definition', 'Positive Examples', 'Negative Examples']
        self.instance_full_task_description[task_name] = {key: json_data_dict[key] for key in task_keys}
        # ----------------- create and format the instructions -----------------
        if "Categories" in json_data_dict:
            json_data_dict["Categories"] = "|||".join(self.tasks[task_name]["Categories"])
        # ----------------- Always add the Task as prefix -----------------
        generic_instruction_all = ""
        generic_instruction = ""
        generic_instruction += "[Task]: %s\n"%task_name.strip()
        generic_instruction_all += "[Task]: %s\n"%task_name.strip()
        # ----------------- Add category -----------------
        if self.data_args.add_category:
            generic_instruction += "[Categories]: %s\n"%(json_data_dict["Categories"])
        generic_instruction_all += "[Categories]: %s\n"%(json_data_dict["Categories"])
        # ----------------- Add task_summary -----------------
        # commenting out as the new way to read is from the meta_data file for tasks and summary is not reqlly used.
        if self.data_args.add_task_summary:
            generic_instruction += "[Summary]: %s\n"%task_summary.strip()
        generic_instruction_all += "[Summary]: %s\n"%task_summary.strip()
        # # TODO: Fix later: sometimes the definition has two newlines
        # ----------------- Add Instruction -----------------
        if type(json_data_dict["Definition"]) == list:
            json_data_dict["Definition"] = " ".join(json_data_dict["Definition"])
        if self.data_args.add_instruction:
            generic_instruction += "[Definition]: %s\n"%json_data_dict["Definition"].strip()
        generic_instruction_all += "[Definition]: %s\n"%json_data_dict["Definition"].strip()
        # ----------------- Add Pos/Neg Examples -----------------
        positive_examples, negative_examples, generic_instruction_all = self.get_pos_neg_examples(
            json_data_dict, generic_instruction_all)
        # ----------------- Create source target pairs for training/evaluation -----------------
        if number_of_instances == -1 or number_of_instances == 1:
            num_instances = len(json_data_dict["Instances"])
        elif number_of_instances < 1.0: # input given as a fraction
            num_instances = int(numpy.ceil(len(json_data_dict["Instances"]) * number_of_instances))
        elif number_of_instances > 1.0: # input given as an absolute value
            num_instances = int(number_of_instances)
        # --------- do not exceed the number of total instances ------------
        num_instances = min(num_instances, len(json_data_dict["Instances"]))
        if self.dataset_mode == self.TRAIN_MODE:
            sources, targets, source_to_targets = self.create_source_target_pairs_for_training(
                json_data_dict, number_of_instances=num_instances)
        if self.dataset_mode == self.EVAL_MODE:
            sources, targets, source_to_targets = self.create_source_target_pairs_for_evaluation(
                json_data_dict, number_of_instances=num_instances) # consider all instances for eval
        # """filter by max length by including all of instructions so that all configs have the same number of items"""
        if self.data_args.filter_dataset_by_length == True:
            sources, targets, source_to_targets, _, _ = self.filter_by_max_length(
                sources, generic_instruction_all, targets, source_to_targets, task_name,
                json_data_dict, positive_examples, negative_examples)

        return generic_instruction, sources, targets, source_to_targets, positive_examples, negative_examples

    def filter_by_max_length(
            self, sources, generic_instruction_all, targets, source_to_targets, task_name, json_data_dict,
            positive_examples, negative_examples):
        """filter by max length by including all of instructions so that all configs have the same number of items"""
        length_filtered_sources = []
        length_filtered_targets = []
        length_filtered_source_to_targets = []
        length_filtered_positive_examples = []
        length_filtered_negative_examples = []

        if len(sources) > 0:
            instruction_source = []
            instruction_target = []
            for index in range(len(sources)):
                if self.training_args.prepend_inst_in_enc_or_dec =="encoder":
                    instruction_source.append(generic_instruction_all + sources[index])
                    instruction_target.append(targets[index])
                if self.training_args.prepend_inst_in_enc_or_dec =="decoder":
                    instruction_source.append(sources[index])
                    instruction_target.append(generic_instruction_all + targets[index])

            tokenized_inputs_sources = self.tokenizer(instruction_source, truncation=False)
            tokenized_inputs_targets = self.tokenizer(instruction_target, truncation=False)

            for index in range(len(sources)):
                if (len(tokenized_inputs_sources[index]) <= self.data_args.max_source_length and
                        len(tokenized_inputs_targets[index]) <= self.data_args.max_target_length):
                    length_filtered_sources.append(sources[index])
                    length_filtered_targets.append(targets[index])
                    length_filtered_source_to_targets.append(source_to_targets[index])

            #  Actual number of source target pairs can exceed instances since some sources have multiple reference targets
            print("Loaded dataset: %s Total: [%d], Selected and Filtered [%d] / [%d]"%(
                task_name, len(json_data_dict['Instances']), len(length_filtered_sources), len(sources)))
            
        return (length_filtered_sources, length_filtered_targets, length_filtered_source_to_targets,
                length_filtered_positive_examples, length_filtered_negative_examples)

    def get_pos_neg_examples(self, json_data_dict, generic_instruction_all):
        # ---------- create positive examples --------------
        positive_examples  = json_data_dict["Positive Examples"]
        positive_examples_list = []
        for example in positive_examples:
            example_string = "%s %s\n%s %s\n"%(self.input_prefix, example["input"].strip(),
                                               self.output_prefix, example["output"].strip())
            if self.data_args.add_explanations:
                example_string += "[Explanation]: %s\n"%example["explanation"].strip()
        positive_examples_list.append(example_string)

        generic_instruction_all += "[Positive Examples]:\n"
        generic_instruction_all = "%s %s\n%s %s\n"%(self.input_prefix, positive_examples[0]["input"].strip(),
                                                    self.output_prefix, positive_examples[0]["output"].strip())
                                            
        # generic_instruction_all += "[Explanation]: %s\n"%positive_examples[0]["explanation"].strip()
        # ---------- create negative examples --------------
        negative_examples  = json_data_dict["Negative Examples"]
        negative_examples_list = []
        for example in negative_examples:
            example_string = "%s %s\n[Negative Output]: %s\n"%(self.input_prefix, example["input"].strip(),
                                                               example["output"].strip())
            if self.data_args.add_explanations:
                example_string += "[Explanation]: %s\n"%example["explanation"].strip()
        negative_examples_list.append(example_string)

        # generic_instruction_all += "[Negative Examples]:\n"
        # generic_instruction_all += "%s %s\n[Negative Output]: %s\n"%(
        #       self.input_prefix, negative_examples[0]["input"].strip(), negative_examples[0]["output"].strip())
        # generic_instruction_all += "[Explanation]: %s\n"%negative_examples[0]["explanation"].strip()

        return positive_examples_list, negative_examples_list, generic_instruction_all

    def create_source_target_pairs_for_evaluation(self, json_data_dict, number_of_instances=-1):
        """
            Since each input can have multiple outputs, we create a single input and a list of outputs
            a pair for input output pair for evaluation. We do not want to predict the source multiple times.
        """
        sources, targets, source_to_targets = [], [], []
        # --------- do not exceed the number of total instances ------------
        number_of_instances = min(number_of_instances, len(json_data_dict["Instances"]))
        # TODO: later select a random set of instances instead of the first
        for index in range(number_of_instances):
            source_str = "%s %s"%(self.input_prefix, json_data_dict['Instances'][index]['input'])
            assert(type(json_data_dict['Instances'][index]['output'])==list)
            sources.append(source_str)
            """
                Note: We only use the first target for evaluation in a batch.
                This would be used for computing the loss.
                We create a separate list which adds all the targets to be used for computing ROUGE type metrics.
            """
            target_strings = ["%s %s"%(self.output_prefix, target)
                              for target in json_data_dict['Instances'][index]['output']]
            assert (type(target_strings==list))

            targets.append(target_strings[0])
            source_to_targets.append(target_strings)
        return sources, targets, source_to_targets

    def create_source_target_pairs_for_training(self, json_data_dict, number_of_instances=-1):
        """
            Since each input can have multiple outputs, we create one input/output pair for each for training.
        """
        sources, targets, source_to_targets = [], [], []
        for index in range(number_of_instances):
            source_str = "%s %s"%(self.input_prefix, json_data_dict['Instances'][index]['input'])
            assert(type(json_data_dict['Instances'][index]['output'])==list)
            for target in json_data_dict['Instances'][index]['output']:
                target_str = "%s %s"%(self.output_prefix, target)
                sources.append(source_str)
                targets.append(target_str)
                assert(self.input_prefix not in target_str)
                assert(self.output_prefix not in source_str)
                source_to_targets.append(target_str)
        return sources, targets, source_to_targets

    def decode_labels_and_predictions(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        if self.data_args.ignore_pad_token_for_loss:
            # ---------- Replace -100 in the labels as we can't decode them. ---------
            labels = numpy.where(labels != -100, labels, self.tokenizer.pad_token_id)
            preds = numpy.where(preds != -100, preds, self.tokenizer.pad_token_id)
        preds = preds.tolist()
        labels = labels.tolist()
        # -------------extract the predictions--------------
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # -------------extract the references---------------
        decoded_labels = self.instances_source_to_targets[0:len(decoded_preds)]
        inputs = self.instances_source[0:len(decoded_preds)]
        # ======================== Some simple post-processing ========================
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        # ======================== Separate into tasks ========================
        decoded_preds_labels_by_tasks = {}
        decoded_preds_labels_indexes_by_tasks = {}
        for index in range(len(self.instances_task_name)):
            decoded_preds_indexes = self.tokenizer(decoded_preds[index], add_special_tokens=False)["input_ids"]
            decoded_label_indexes = self.tokenizer(decoded_labels[index], add_special_tokens=False)["input_ids"]
            if self.instances_task_name[index] not in decoded_preds_labels_by_tasks:
                # ----------encode the cleaned text------------
                decoded_preds_labels_by_tasks[self.instances_task_name[index]] = {"preds": [decoded_preds[index]], "labels":[decoded_labels[index]], "inputs":[inputs[index]]}
                decoded_preds_labels_indexes_by_tasks[self.instances_task_name[index]] = {"preds": [decoded_preds_indexes], "labels":[decoded_label_indexes]}
            else:
                decoded_preds_labels_by_tasks[self.instances_task_name[index]]["preds"].append(decoded_preds[index])
                decoded_preds_labels_by_tasks[self.instances_task_name[index]]["labels"].append(decoded_labels[index])
                decoded_preds_labels_by_tasks[self.instances_task_name[index]]["inputs"].append(inputs[index])
                
                decoded_preds_labels_indexes_by_tasks[self.instances_task_name[index]]["preds"].append(decoded_preds_indexes)
                decoded_preds_labels_indexes_by_tasks[self.instances_task_name[index]]["labels"].append(decoded_label_indexes)

        return decoded_preds, decoded_labels, decoded_preds_labels_by_tasks, decoded_preds_labels_indexes_by_tasks

    def postprocess_text(self, preds, labels):
        """Cleans model output to leave only generated text.

        Removes the output prefix from the decoded string
        Removes trailing spaces
        Adds newline at sentence boundaries as rougeLSum expects newline after each sentence
        """
        preds_cleaned_list = []
        for pred_string in preds:
            # if self.output_prefix in pred_string:
            # -------- take the last part of the split for the actual output ---------
            pred_string = pred_string.split(self.output_prefix)
            expected_number_of_splits = 1 + self.data_args.add_positive_examples + self.data_args.add_negative_examples
            if len(pred_string) >= expected_number_of_splits: # The model was able to fully decode with all the prefixes
                pred_string = pred_string[-1].strip()
            else: # model did not have the neceesary number of splits probably due exceeding the decode length
                pred_string = ""
            pred_string = " ".join(pred_string.strip().split())
            pred_string = "\n".join(nltk.sent_tokenize(pred_string.strip()))
            preds_cleaned_list.append(pred_string)
        # ---------- labels are list of lists so process accordingly ----------
        edited_labels_list = []
        for labels_per_source in labels:
            # -------- take the last part of the split for the actual output ---------
            labels_per_source = [label.split(self.output_prefix)[-1].strip() for label in labels_per_source]
            labels_per_source = [" ".join(label.strip().split()) for label in labels_per_source]
            labels_per_source = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels_per_source]
            edited_labels_list.append(labels_per_source)
        return preds_cleaned_list, edited_labels_list
