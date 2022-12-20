import collections
import random
import pdb

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class FewShotBatchSampler:
    def __init__(self, dataset_tasks_list=None, n_way=2, k_shot=4, include_query=False, shuffle=True,
                 shuffle_once=False, task_sampling_mode="uniform", target_tasks=None,
                 target_task_sampling_rate_increase=0.0):
        """
        Args:
            dataset_tasks_list - PyTorch tensor of the labels [tasks] of the data elements.
            n_way - Number of classes[tasks] to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size n_way*k_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                    iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                        the beginning, but kept constant across iterations
                        (for validation)
        """
        super().__init__()
        self.max_task_probability = 0.5
        self.dataset_tasks_list = dataset_tasks_list
        self.n_way = n_way
        # assert self.n_way % 2 == 0, "Please set --num_tasks_per_iter as a multiple 2"
        self.k_shot = k_shot
        self.shuffle = shuffle
        self.include_query = include_query
        self.task_sampling_mode = task_sampling_mode
        if self.include_query:
            self.k_shot *= 2
        self.batch_size = self.n_way * self.k_shot  # Number of overall items per batch
        # ------------ Organize examples by class ------------
        self.classes = torch.unique(self.dataset_tasks_list).tolist()
        if target_tasks is not None:
            self.target_tasks = np.unique(target_tasks).tolist()
        else:
            self.target_tasks = None
        self.target_task_sampling_rate_increase =  target_task_sampling_rate_increase
        # ------------ Compute task sampling probabilities ------------
        self.dataset_task_freq = collections.Counter(dataset_tasks_list.tolist())
        for k, v in self.dataset_task_freq.items():
            self.dataset_task_freq[k] = v / len(dataset_tasks_list)
        if self.task_sampling_mode == "proportionate":
            self.class_probabilities = [self.dataset_task_freq[c] for c in self.classes]
        if self.task_sampling_mode == "uniform":
            self.class_probabilities = np.ones(len(self.classes)) / len(self.classes)
        # ------------ Set the indices per class ------------
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_tasks_list == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot
        # ------------ Create a list of classes from which we select the N classes per batch ------------
        self.iterations = sum(self.batches_per_class.values()) // self.n_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # ------------ For testing, we iterate over classes instead of shuffling them ------------
            sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def increment_target_task_sampling_rates(self, class_probabilities):
        #  linearly increase the rate for target tasks until a max value
        for task in self.target_tasks:
            if task in self.classes:
                task_index  = self.classes.index(task)
                # -------- increment till it reaches a max -----------
                if class_probabilities[task_index] < self.max_task_probability:
                    class_probabilities[task_index] += self.target_task_sampling_rate_increase
        # -------- normalize them back ------------
        class_probabilities = class_probabilities / sum(class_probabilities)
        return class_probabilities

    def get_target_test_sampling_rate(self):
        target_task_probabilites = {}
        for task in self.target_tasks:
            if task in self.classes:
                task_index  = self.classes.index(task)
                target_task_probabilites[task] = self.class_probabilities[task_index]
                return target_task_probabilites
            else:
                return None

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()
        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            # --------- sample unique tasks -----------
            class_batch = [-1] * self.n_way
            if self.n_way >= 2:
                # find unique tasks in batches so that source and target tasks are different.
                while len(class_batch) != len(set(class_batch)):
                    class_batch = np.random.choice(self.classes, self.n_way, p=self.class_probabilities).tolist()
            else:
                class_batch = np.random.choice(self.classes, self.n_way, p=self.class_probabilities).tolist()


            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                start = start_index[c]
                end = start_index[c] + self.k_shot
                batch_items = self.indices_per_class[c][start:end]
                index_batch.extend(batch_items)
                start_index[c] += self.k_shot
                # --------- need to wrap the indexes back as the next iter would go over --------------
                if start_index[c] + self.k_shot >= len(self.indices_per_class[c]):
                    start_index[c] = 0
            # --------- increment the target_task sampling rates ---------
            if self.target_tasks is not None and self.target_task_sampling_rate_increase>0:
                self.class_probabilities = self.increment_target_task_sampling_rates(self.class_probabilities)

            yield index_batch

    def __len__(self):
        return self.iterations

@dataclass
class DataCollatorForSeq2SeqForMaml(DataCollatorForSeq2Seq):
    """
        Data collator that will dynamically pad the inputs received, as well as the labels.

        Args:
            tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
                The tokenizer used for encoding the data.
            model (:class:`~transformers.PreTrainedModel`):
                The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
                prepare the `decoder_input_ids`

                This is useful when using `label_smoothing` to avoid calculating loss twice.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence is provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
            label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
                The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: None
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, list_of_features, return_tensors=None):
        """
            Args:
                - list_of features is a list with length = 1
                - list_of features[0], is a dict with key = task_id
                    - there are num_tasks_per_iter tasks with task_ids as the key
                - list_of features[0][task_id] is a list of dicts
                - list_of features[0][task_id][index] is a dict
                    - with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                    - has list of tensors of length task_batch_size_per_iter
            Returns:
                - return_list_of_features: dict of dicts,
                    - with task_id as key
                    - has task_batch_size_per_iter # of tasks
                - return_list_of_features[task_id] is a dict
                    -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                - return_list_of_features[task_id][key] is a tensor of length task_batch_size_per_iter
        """
        list_of_features = list_of_features[0] # batch size is assumed to 1 for this, and the features internally have the n_way, k_shot batch
        return_list_of_features = {}
        if return_tensors is None:
            return_tensors = self.return_tensors
        for key, features in list_of_features.items():
            labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the same length to return tensors.
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    feature["labels"] = (feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"])
            # We have to pad the decoder_input_ids before calling `tokenizer.pad` as this method won't pad them and needs them of the same length to return tensors.
            decoder_input_ids = [feature["decoder_input_ids"] for feature in features] if "decoder_input_ids" in features[0].keys() else None
            if decoder_input_ids is not None:
                max_label_length = max(len(l) for l in decoder_input_ids)
                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature["decoder_input_ids"]))
                    feature["decoder_input_ids"] = (feature["decoder_input_ids"] + remainder if padding_side == "right" else remainder + feature["decoder_input_ids"])
            # -------------- now pad all the features to the constant lenght -------------------
            features = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=return_tensors,)
            # ------------- prepare decoder_input_ids -------------
            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
                features["decoder_input_ids"] = decoder_input_ids
            return_list_of_features[key] = features
        keys = list(return_list_of_features.keys())
        assert len(keys) == len(set(keys))
        # assert len(keys) % 2 == 0, "%s please set --num_tasks_per_iter as a multiple of 2"%keys

        return return_list_of_features

    def pad_fields(self, features, input_key, pad_token):
        input_list = [feature[input_key] for feature in features] if input_key in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if input_list is not None:
            max_label_length = max(len(l) for l in input_list)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [pad_token] * (max_label_length - len(feature[input_key]))
                feature[input_key] = (feature[input_key] + remainder if padding_side == "right" else remainder + feature[input_key])

@dataclass
class DataCollatorForCausalLanguageModeling:
    """
        Data collator used for language modeling.
        - collates batches of tensors, honoring their tokenizer's pad_token
        - preprocesses batches for masked language modeling
    """
    tokenizer: None
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (feature["labels"] + remainder if padding_side == "right" else remainder + ["labels"])
        #  -------------- remove None entries ---------------
        features = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length, return_tensors=return_tensors)
        
        return features

@dataclass
class DataCollatorForS2SHnet:
    """
        Since this has additional inputs, need to pad them separately

        Data collator that will dynamically pad the inputs received, as well as the labels.

        Args:
            tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
                The tokenizer used for encoding the data.
            model (:class:`~transformers.PreTrainedModel`):
                The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
                prepare the `decoder_input_ids`

                This is useful when using `label_smoothing` to avoid calculating loss twice.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence is provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
            label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
                The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: None
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def pad_fields(self, features, input_key, pad_token):
        input_list = [feature[input_key] for feature in features] if input_key in features[0].keys() else None
        # same length to return tensors.
        if input_list is not None:
            max_label_length = max(len(l) for l in input_list)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [pad_token] * (max_label_length - len(feature[input_key]))
                feature[input_key] = (feature[input_key] + remainder if padding_side == "right" else remainder + feature[input_key])

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        """
            We have to pad the features before calling `tokenizer.pad` as this method won't pad them and needs them of the same length to return tensors.
        """
        self.pad_fields(features, "labels", self.label_pad_token_id)
        self.pad_fields(features, "input_ids_hnet", 1)
        self.pad_fields(features, "attention_mask_hnet", 0)
        features = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length,pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=return_tensors)
        # ----------------- prepare decoder_input_ids -----------------
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class DataCollatorForSeq2SeqForHNETMaml(DataCollatorForSeq2Seq):
    """
        Data collator that will dynamically pad the inputs received, as well as the labels.

        Args:
            tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
                The tokenizer used for encoding the data.
            model (:class:`~transformers.PreTrainedModel`):
                The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
                prepare the `decoder_input_ids`

                This is useful when using `label_smoothing` to avoid calculating loss twice.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence is provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
            label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
                The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: None
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def pad_fields(self, features, input_key, pad_token):
        input_list = [feature[input_key] for feature in features] if input_key in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if input_list is not None:
            max_label_length = max(len(l) for l in input_list)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [pad_token] * (max_label_length - len(feature[input_key]))
                feature[input_key] = (feature[input_key] + remainder if padding_side == "right" else remainder + feature[input_key])

    def __call__(self, list_of_features, return_tensors=None):
        """
            Args:
                - list_of features is a list with length = 1
                - list_of features[0], is a dict with key = task_id
                    - there are num_tasks_per_iter tasks with task_ids as the key
                - list_of features[0][task_id] is a list of dicts
                - list_of features[0][task_id][index] is a dict
                    - with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                    - has list of tensors of length task_batch_size_per_iter
            Returns:
                - return_list_of_features: dict of dicts,
                    - with task_id as key
                    - has task_batch_size_per_iter # of tasks
                - return_list_of_features[task_id] is a dict
                    -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                - return_list_of_features[task_id][key] is a tensor of length task_batch_size_per_iter
        """
        list_of_features = list_of_features[0] # batch size is assumed to 1 for this, and the features internally have the n_way, k_shot batch
        return_list_of_features = {}
        if return_tensors is None:
            return_tensors = self.return_tensors
        for key, features in list_of_features.items():
            self.pad_fields(features, "labels", self.label_pad_token_id)
            self.pad_fields(features, "input_ids_hnet", 1)
            self.pad_fields(features, "attention_mask_hnet", 0)
            # -------------- now pad all the features to the constant lenght -------------------
            features = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=return_tensors,)
            # ------------- prepare decoder_input_ids -------------
            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
                features["decoder_input_ids"] = decoder_input_ids
            return_list_of_features[key] = features
        keys = list(return_list_of_features.keys())
        assert len(keys) == len(set(keys))
        # assert len(keys) % 2 == 0, "%s please set --num_tasks_per_iter as a multiple of 2"%keys

        return return_list_of_features
    
    def pad_fields(self, features, input_key, pad_token):
        input_list = [feature[input_key] for feature in features] if input_key in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if input_list is not None:
            max_label_length = max(len(l) for l in input_list)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [pad_token] * (max_label_length - len(feature[input_key]))
                feature[input_key] = (feature[input_key] + remainder if padding_side == "right" else remainder + feature[input_key])

class AbstractNIDataHandler(Dataset):
    """
        Interface of DataHanlders for Natural Interactions tasks.
        When using for training, subclasses must implement methods to populate attributes:
            * self.train_dataset
            * self.dev_dataset
            * self.test_dataset

        Attributes:
            train_dataset: torch.Dataset instance. Must be populated by subclass.
            dev_dataset: torch.Dataset instance. Must be populated by subclass.
            test_dataset: torch.Dataset instance. Must be populated by subclass.
            data_collator:
            batch_sampler_train:
            batch_sampler_eval:
    """
    def __init__(self, model_config, model_args, training_args, data_args, logger, tokenizer, model,
                 return_tensors="pt"):
        self.model_config =  model_config
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataset_folder = data_args.dataset_folder
        self.return_tensors = return_tensors
        self.model = model

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
    
    def create_batch_samplers_and_collators(self):
        # Create a Batch Sampler for Standard and Meta-Learning loops
        # This needs to be called after the datasets have been instantiated for MAML loop.
        self.label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if self.training_args.train_loop_opt_type == "standard":
            if self.training_args.run_hnet_in_batch_mode_per_task:
                self.batch_sampler_train = FewShotBatchSampler(
                dataset_tasks_list=torch.LongTensor(self.train_dataset.instances_task_indexes),
                n_way=self.data_args.num_tasks_per_iter, k_shot=self.data_args.task_batch_size_per_iter,
                include_query=False, shuffle=True, shuffle_once=False,
                task_sampling_mode=self.data_args.task_sampling_mode, target_tasks=self.test_dataset.instances_task_indexes,
                target_task_sampling_rate_increase=self.training_args.target_task_sampling_rate_increase)

                self.batch_sampler_eval = None
                self.data_collator = DataCollatorForSeq2SeqForMaml(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
                self.data_collator_eval = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
            else:
                self.batch_sampler_train = None
                self.batch_sampler_eval = None
                self.data_collator = DataCollatorForSeq2Seq(self.model.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id,pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
                self.data_collator_eval = DataCollatorForSeq2Seq(self.model.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id,pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)

        if self.training_args.train_loop_opt_type in ["maml"]:
            self.batch_sampler_train = FewShotBatchSampler(
                dataset_tasks_list=torch.LongTensor(self.train_dataset.instances_task_indexes),
                n_way=self.data_args.num_tasks_per_iter, k_shot=self.data_args.task_batch_size_per_iter,
                include_query=False, shuffle=True, shuffle_once=False,
                task_sampling_mode=self.data_args.task_sampling_mode, target_tasks=self.test_dataset.instances_task_indexes,
                target_task_sampling_rate_increase=self.training_args.target_task_sampling_rate_increase)

            self.batch_sampler_eval = None
            self.data_collator = DataCollatorForSeq2SeqForMaml(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
            self.data_collator_eval = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
        
        if self.training_args.train_loop_opt_type == "hnet":
            if self.training_args.run_hnet_in_batch_mode_per_task:
                #  This uses the same batch sampler as the MAML which samples per task.
                self.batch_sampler_train = FewShotBatchSampler(
                dataset_tasks_list=torch.LongTensor(self.train_dataset.instances_task_indexes),
                n_way=self.data_args.num_tasks_per_iter, k_shot=self.data_args.task_batch_size_per_iter,
                include_query=False, shuffle=True, shuffle_once=False,
                task_sampling_mode=self.data_args.task_sampling_mode, target_tasks=self.test_dataset.instances_task_indexes,
                target_task_sampling_rate_increase=self.training_args.target_task_sampling_rate_increase)
                
                self.data_collator = DataCollatorForSeq2SeqForHNETMaml(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)

                self.data_collator_eval = DataCollatorForS2SHnet(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
                self.batch_sampler_eval = None
            else:
                self.batch_sampler_train = None
                self.batch_sampler_eval = None
                self.data_collator = DataCollatorForS2SHnet(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
                self.data_collator_eval = DataCollatorForS2SHnet(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)

        if self.training_args.train_loop_opt_type in ["hnet_maml"]:
            self.batch_sampler_train = FewShotBatchSampler(
                dataset_tasks_list=torch.LongTensor(self.train_dataset.instances_task_indexes),
                n_way=self.data_args.num_tasks_per_iter, k_shot=self.data_args.task_batch_size_per_iter,
                include_query=False, shuffle=True, shuffle_once=False,
                task_sampling_mode=self.data_args.task_sampling_mode, target_tasks=self.test_dataset.instances_task_indexes,
                target_task_sampling_rate_increase=self.training_args.target_task_sampling_rate_increase)
            self.batch_sampler_eval = None
            
            self.data_collator = DataCollatorForSeq2SeqForHNETMaml(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
            
            self.data_collator_eval = DataCollatorForS2SHnet(self.tokenizer, model=self.model, label_pad_token_id=self.label_pad_token_id, pad_to_multiple_of=8 if self.training_args.fp16 else None, return_tensors=self.return_tensors)
                
    def get_data_collator(self, model, train_loop_opt_type="standard", fp16=False, label_pad_token_id=-100,
                          return_tensors="pt"):
        if ("bart" in self.model_args.model_name_or_path or "allenai/led" in self.model_args.model_name_or_path):
            if train_loop_opt_type in ["maml"]:
                return DataCollatorForSeq2SeqForMaml(
                    model.tokenizer, model, label_pad_token_id=label_pad_token_id,
                    pad_to_multiple_of=8 if fp16 else None, return_tensors=return_tensors)
            if train_loop_opt_type == "standard":
                return DataCollatorForSeq2Seq(
                    model.tokenizer, model=model, label_pad_token_id=label_pad_token_id,
                    pad_to_multiple_of=8 if fp16 else None, return_tensors=return_tensors)
            if train_loop_opt_type == "hnet":
                return DataCollatorForS2SHnet(
                    model.tokenizer, model=model, label_pad_token_id=label_pad_token_id,
                    pad_to_multiple_of=8 if fp16 else None, return_tensors=return_tensors)
            
            if train_loop_opt_type == "hnet_maml":
                return DataCollatorForSeq2SeqForHNETMaml(
                    model.tokenizer, model=model, label_pad_token_id=label_pad_token_id,
                    pad_to_multiple_of=8 if fp16 else None, return_tensors=return_tensors)

        if "gpt2" in self.model_args.model_name_or_path:
            return DataCollatorForCausalLanguageModeling(model.tokenizer)        

    def get_target_test_sampling_rate(self):
        if self.training_args.train_loop_opt_type in ["standard", "hnet"]:
            return None
        if self.training_args.train_loop_opt_type in ["maml", "hnet_maml"]:
            return self.batch_sampler_train.get_target_test_sampling_rate()

    def print_sample_predictions(self, decoded_preds_labels_by_tasks):
        raise NotImplementedError
