
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import pdb
from copy import deepcopy
from tqdm.auto import tqdm



# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.optimization import AdamW

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)


from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    IterableDatasetShard,
    LengthGroupedSampler,
    get_parameter_names,
)
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer
from transformers.trainer import is_torch_tpu_available
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_utils import SchedulerType
import datasets

import learn2learn as l2l
# from torchviz import make_dot, make_dot_from_trace

from packaging import version
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)

from .models_meta import MAMLS2SModel

class DerivedTrainerClass(Seq2SeqTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            data_collator_eval: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            predict_dataset_val: Optional[Dataset] = None,
            predict_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            batch_sampler_train=None,
            batch_sampler_eval=None):

        super().__init__(model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers
        )
        self.batch_sampler=batch_sampler_train
        self.batch_sampler_eval=batch_sampler_eval
        self.data_collator_eval = data_collator_eval
        self.predict_dataset = predict_dataset
        self.predict_dataset_val = predict_dataset_val

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        sch_name = SchedulerType(self.args.lr_scheduler_type)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[sch_name]
        print("Creating %s Scheduler using %s"%(sch_name, schedule_func))
        if sch_name == SchedulerType.CONSTANT:
            self.lr_scheduler = schedule_func(optimizer)
        # ------------ All other schedulers require `num_warmup_steps`
        elif self.args.warmup_steps is None:
            raise ValueError(f"{sch_name} requires `num_warmup_steps`, please provide that argument.")
        elif sch_name == SchedulerType.CONSTANT_WITH_WARMUP:
            self.lr_scheduler = schedule_func(optimizer, num_warmup_steps=self.args.warmup_steps)
        # ------------ All other schedulers require `num_training_steps`
        elif num_training_steps is None:
            self.lr_scheduler = ValueError(f"{sch_name} requires `num_training_steps`, please provide that argument.")
        elif sch_name == SchedulerType.COSINE_WITH_RESTARTS or sch_name == SchedulerType.COSINE:
            self.lr_scheduler = schedule_func(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps, num_cycles = self.args.num_cycles_cosine_schd)
        elif sch_name == SchedulerType.POLYNOMIAL:
            self.lr_scheduler = schedule_func(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps, power = self.args.power_polynomial_schd)
        else:
            self.lr_scheduler = schedule_func(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps)
        return self.lr_scheduler

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        # by default the save_checkpoint_overide is false, needs to be set to true if we plan to save checkpoints. 
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)
            
        if self.control.should_save and self.args.save_checkpoints:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def save_model(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = self.args.output_dir

        super().save_model(output_dir)

        # Save extra arguments necessary to create model instance
        if hasattr(self.model, "save_extra_args"):
            self.model.save_extra_args(output_dir)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None
        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if self.batch_sampler:
                    return self.batch_sampler
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset, generator=generator)
                return RandomSampler(self.train_dataset)

            elif (self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL] and not self.args.dataloader_drop_last):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

    def get_train_dataloader(self) -> DataLoader:
        """
            Returns the training :class:`~torch.utils.data.DataLoader`.

            Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
            to distributed training if necessary) otherwise.

            Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        # psb.set_trace()
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:# We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
            ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        args = self.args
        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn("`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.", FutureWarning,)
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        self._hp_search_setup(trial)# This might change the seed so needs to run first.

        model_reloaded = False
        if self.model_init is not None: # Model re-init
            set_seed(args.seed) # Seed must be set before instantiating the model when using model_init.
            self.model = self.call_model_init(trial)
            model_reloaded = True
            self.optimizer, self.lr_scheduler = None, None # Reinitializes optimizer and scheduler

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                # raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
                print(f"No valid checkpoint found in output directory. Starting from scratch ({args.output_dir})")
        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                # raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
                print(f"Can't find a valid checkpoint at {resume_from_checkpoint} as os path %s does not exist"%os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
            else:
                logger.info(f"Loading model from {resume_from_checkpoint}).")
                if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                    config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                    checkpoint_version = config.transformers_version
                    if checkpoint_version is not None and checkpoint_version != __version__:
                        logger.warn(
                            f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                            f"Transformers but your current version is {__version__}. This is not recommended and could "
                            "yield to errors or unwanted behaviors.")

                if args.deepspeed: # will be resumed in deepspeed_init
                    pass
                else: # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                    self._load_state_dict_in_model(state_dict) # If the model is on the GPU, it still works!
                    # release memory
                    del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # ------------------ Train! ------------------
        num_examples = (self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (((step + 1) % args.gradient_accumulation_steps != 0) and args.local_rank != -1 and args._no_sync_in_gradient_accumulation):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                # last step in epoch but step is always smaller than gradient_accumulation_steps
                if (step + 1) % args.gradient_accumulation_steps == 0 or (steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed: # deepspeed does its own clipping

                        if self.use_amp:# AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):# Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):# Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:# Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(amp.master_params(self.optimizer) if self.use_apex else model.parameters(), args.max_grad_norm,)
                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    
                    if self.control.should_training_stop ==True:
                        print("Training stopped due to max steps reached: %d %s "%(self.state.global_step, self.state.max_steps))

                    # if args.local_rank in [0, -1]:
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            # TODO: set this through params
            # self.control.should_evaluate = evaluate_metrics_after_epoch
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            
            # if args.local_rank in [0, -1]:
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            print("==============================================================")
            logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )
        else:
            logger.info(f"Loading last chekcpoint from {get_last_checkpoint(args.output_dir)} (score: {self.state.best_metric}).")

        """
            Reevaluate all the metrics for either the best, or the last checkpoint for reporting
            If the load_best_model_at_end is set to None
        """
        # ------------set the num_eval_items to max for final eval---------------
        metrics = self.evaluate(eval_dataset=self.predict_dataset_val)
        print("Final metrics for validation tasks on the eval split")
        print("Metrics from checkpoint: %s"%self.state.best_model_checkpoint)
        print("==============================================================")
        for k, v in metrics.items():
            print("%s: %s"%(k, v))
        print("==============================================================")

        print("Final metrics for eval tasks on the eval split")
        metrics = self.evaluate(eval_dataset=self.predict_dataset)
        print("Metrics from checkpoint: %s"%self.state.best_model_checkpoint)
        print("==============================================================")
        for k, v in metrics.items():
            print("%s: %s"%(k, v))
        print("==============================================================")

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.run_hnet_in_batch_mode_per_task:
            task_list = list(inputs.keys())
            losses = []
            for index, task in enumerate(task_list):
                input_task = inputs[task]
                if self.use_amp:
                    with autocast():
                        loss = self.compute_loss(model, input_task)
                else:
                    loss = self.compute_loss(model, input_task)
                losses.append(loss)
            loss = torch.mean(torch.vstack(losses))
        else:
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), dataloader=dataloader)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
   
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None,) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Perform an evaluation step on :obj:`model` using obj:`inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (:obj:`nn.Module`):
                    The model to evaluate.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (:obj:`bool`):
                    Whether or not to return the loss only.

            Return:
                Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
                labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step( model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        """
            Need to make sure that decoder_input_ids is set in get_kwargs
            This should be available in the inputs
        """
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "decoder_input_ids": inputs['decoder_input_ids']
        }

        generated_tokens = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs,)
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator_eval,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator_eval,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            else:
                from ray import tune

                run_id = tune.get_trial_id()
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        # if is_torch_tpu_available():
        #     xm.rendezvous("saving_optimizer_states")
        #     xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        # elif is_sagemaker_mp_enabled():
        #     if smp.dp_rank() == 0:
        #         # Consolidate the state dict on all processed of dp_rank 0
        #         opt_state_dict = self.optimizer.state_dict()
        #         # Save it and the scheduler on the main process
        #         if self.args.should_save:
        #             torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
        #             with warnings.catch_warnings(record=True) as caught_warnings:
        #                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #             reissue_pt_warnings(caught_warnings)
        #             if self.use_amp:
        #                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.use_amp:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # if is_torch_tpu_available():
        #     rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class TrainerClassMAML(DerivedTrainerClass):

    def _prepare_inputs_for_train(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        We override the standard one as this one is nested and needs to be kept nested for further processing in the MAML
            Inputs:
                - inputs: dict of dicts,
                    - with task_id as key
                    - has task_batch_size_per_iter # of tasks
                - inputs[task_id] is a dict
                    -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                - inputs[task_id][key] is a tensor of length task_batch_size_per_iter
            Returns:
                input_dict: a dict wrapping the nested dict to make sure the inputs to the forward is general
                    - input_dict["inputs"]: dict of dicts,
                        - with task_id as key
                        - has task_batch_size_per_iter # of tasks
                    - input_dict["inputs"][task_id] is a dict
                        -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                    - input_dict["inputs"][task_id][key] is a tensor of length task_batch_size_per_iter
            """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        # -------- wrap the return dict within another dict to keep forward args to a single param -----------
        input_dict = {}
        input_dict["inputs"] = inputs
        return input_dict

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        This is essentially running the outer loop of MAML while the model.forward is running the inner loop of MAML
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs_for_train(inputs)
        # ---------- call the inner loop of MAML --------------
        if self.use_amp:
            with autocast():
                loss = self.maml_inner_loop(**inputs, model=model)
        else:
            # loss = self.maml_inner_loop(**inputs, model=model)
            loss = self.maml_inner_loop(**inputs, model=model)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()

    def clone_model(self, model):
        meta_model = deepcopy(model)
        self._move_model_to_device(model, self.args.device)
        meta_model = self._wrap_model(meta_model)
        meta_model = meta_model.train()
        meta_params = meta_model.named_parameters()
        return meta_model, meta_params

    def sync_autograd(self, loss, net, retain_graph=False): # DDP and AMP compatible autograd
        if self.args.n_gpu ==1: # single GPU
            grads = torch.autograd.grad(loss, net.parameters())
        else:
            # distributed, with AMP optionally
            net.zero_grad()
            loss.backward(retain_graph=retain_graph)

            # if self.args.use_amp: # PyTorch DDP
            #     loss.backward(retain_graph=retain_graph)
            # else: # Apex DDP
            #     with apex.amp.scale_loss(loss, opt) as scaled_loss:
            #         scaled_loss.backward(retain_graph=retain_graph)
            # # this assumed loss scale is 1 as when it's scaled p.grad might not be the valid grad values!
        grads_2 = [p.grad for p in net.parameters()]
        return grads

    def get_back(self, var_grad_fn):
        print(var_grad_fn)
        for var in var_grad_fn.next_functions:
            if var[0]:
                try:
                    tensor = getattr(var[0], 'variable')
                    print(' - gradient:', tensor.grad)
                except AttributeError as e:
                    self.get_back(var[0])

    def maml_inner_loop(self, inputs=None, model=None):
        """
            Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
            The outer loop is run through the Trainer class, while the forward runs an inner loop
            Arguments:
                - inputs: dict of dicts,
                    - with task_id as key
                    - has task_batch_size_per_iter # of tasks
                        - inputs[task_id] is a dict
                            -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                            - inputs[task_id][key] is a tensor of length task_batch_size_per_iter
                - global_step
                - epoch
            Returns:
                losses
                per_task_target_preds
        """
        total_losses = []
        task_list = list(inputs.keys())
        assert len(task_list) % 2 == 0, "%s please set --num_tasks_per_iter as a multiple of 2"%task_list
        support_tasks = task_list[0: int(len(task_list)/2)]
        target_tasks = task_list[int(len(task_list)/2):]
        # for task_id, input_dict in inputs.items():
        for task_index in range(len(support_tasks)):
            task_losses = []
            if self.args.use_multi_step_loss_optimization:
                per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            # ---------- create a copy of main model and optimizer for inner loop -----------
            meta_learner = model.clone()
            if self.args.n_gpu > 1:
                meta_learner = nn.DataParallel(meta_learner)
            meta_learner = meta_learner.train()
            # --------- split inputs to support and target sets ------------
            # input_dict_support, input_dict_target = self.split_batch_to_support_target(input_dict)
            input_dict_support = inputs[support_tasks[task_index]]
            input_dict_target = inputs[target_tasks[task_index]]
            # ----------- inner loop steps ------------
            for num_step in range(self.args.num_inner_training_steps_per_iter):
                # --------- compute loss and update params on support set ---------
                support_loss, outputs = self.compute_loss(meta_learner, input_dict_support, return_outputs=True)
                if self.args.n_gpu > 1:
                    meta_learner.module.adapt(support_loss)
                else:
                    meta_learner.adapt(support_loss)
                # ---------- compute loss on target set ------------
                if self.args.use_multi_step_loss_optimization and self.state.global_step < self.args.multi_step_loss_num_steps:
                    # meta_learner.config.gradient_checkpointing = True
                    target_loss, outputs = self.compute_loss(meta_learner, input_dict_target, return_outputs=True)
                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                else:
                    if num_step == (self.args.num_inner_training_steps_per_iter - 1):
                        # meta_learner.config.gradient_checkpointing = True
                        target_loss, outputs = self.compute_loss(meta_learner, input_dict_target, return_outputs=True)
                        task_losses.append(target_loss)
            # -------------- End of inner steps. Compute losses ----------------
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            del meta_learner
        # ------------ Compute loss across tasks and return to outer loop ------------
        loss = torch.mean(torch.stack(total_losses))
        return loss

    def split_batch_to_support_target(self, input_dict):
        input_dict_support = {}
        input_dict_target = {}
        for k, v in input_dict.items():
            assert( (v.shape[0] % 2) == 0)
            split_point = int(v.shape[0] / 2)
            input_dict_support[k] = v[0:split_point]
            input_dict_target[k] = v[split_point:]
        return input_dict_support, input_dict_target

    def compute_loss(self, model, input_dict, return_outputs=False):
        """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in input_dict:
            labels = input_dict.pop("labels")
        else:
            labels = None
        outputs = model(**input_dict)

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        return (loss, outputs) if return_outputs else loss

    def loss_backward_optimizer_step(self, model, loss):
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(loss).backward(create_graph=self.args.use_second_order_gradients)
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward(create_graph=self.args.use_second_order_gradients)
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward(create_graph=self.args.use_second_order_gradients)
        # ---------- Gradient clipping ----------
        if self.args.max_grad_norm_meta is not None and self.args.max_grad_norm_meta > 0 and not self.deepspeed: # deepspeed does its own clipping
            if self.use_amp:# AMP: gradients need unscaling
                self.scaler.unscale_(self.inner_optimizer)
            if hasattr(self.inner_optimizer, "clip_grad_norm"):# Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                self.inner_optimizer.clip_grad_norm(self.args.max_grad_norm_meta)
            elif hasattr(model, "clip_grad_norm_"):# Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                model.clip_grad_norm_(self.args.max_grad_norm_meta)
            else:# Revert to normal clipping otherwise, handling Apex or full precision
                nn.utils.clip_grad_norm_(amp.master_params(self.inner_optimizer) if self.use_apex else model.parameters(), self.args.max_grad_norm_meta,)
        # ---------- Optimizer step ----------
        optimizer_was_run = True
        if self.deepspeed:
            pass  # called outside the loop
        elif is_torch_tpu_available():
            xm.optimizer_step(self.inner_optimizer)
        elif self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.inner_optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.inner_optimizer.step()

    def create_inner_optimizer_and_scheduler(self, model, learning_rate):
        # if self.optimizer is None:
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = learning_rate
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return optimizer

    def get_per_step_loss_importance_vector(self):
        """
            Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
            loss towards the optimization loss.
            :return: A tensor to be used to compute the weighted average of the loss, useful for
            the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.num_inner_training_steps_per_iter)) * (1.0 / self.args.num_inner_training_steps_per_iter)

        decay_rate = 1.0 / self.args.num_inner_training_steps_per_iter / self.args.multi_step_loss_num_steps
        min_value_for_non_final_losses = 0.03 / self.args.num_inner_training_steps_per_iter

        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.state.global_step * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(loss_weights[-1] + (self.state.global_step * (self.args.num_inner_training_steps_per_iter - 1) * decay_rate), 1.0 - ((self.args.num_inner_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.args.device)
        return loss_weights

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(), create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)


        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {name.replace('module.', ''): value.unsqueeze(0).repeat(
                                    [num_devices] + [1 for i in range(len(value.shape))]) for name, value in names_weights_copy.items()}


        return names_weights_copy

class TrainerClassHNET(DerivedTrainerClass):
    """
        Adapted from Code for Editing Factual Knowledge in Language Models (https://arxiv.org/abs/2104.08164). 
        https://github.com/nicola-decao/KnowledgeEditor
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            data_collator_eval: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            predict_dataset_val: Optional[Dataset] = None,
            predict_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            batch_sampler_train=None,
            batch_sampler_eval=None):

        super().__init__(model=model,
                        args=args,
                        data_collator=data_collator,
                        data_collator_eval=data_collator_eval,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        predict_dataset_val=predict_dataset_val,
                        predict_dataset=predict_dataset,
                        tokenizer=tokenizer,
                        model_init=model_init,
                        compute_metrics=compute_metrics,
                        callbacks=callbacks,
                        optimizers=optimizers,
                        batch_sampler_train=batch_sampler_train,
                        batch_sampler_eval=batch_sampler_eval)

        self.batch_sampler=batch_sampler_train
        self.batch_sampler_eval=batch_sampler_eval
        self.data_collator_eval = data_collator_eval
        self.predict_dataset = predict_dataset
        self.predict_dataset_val = predict_dataset_val

        if self.args.hnet_initial_alt_mode =="hnet":
            self.toggle_train_forward_mode = "hnet"
        if self.args.hnet_initial_alt_mode == "main_lm":
            self.toggle_train_forward_mode = "main_lm"
        
        self.train_steps = 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
            Perform a training step on a batch of inputs.
            This is essentially running the outer loop of MAML while the model.forward is running the inner loop of MAML
            Args:
                model (:obj:`nn.Module`):
                    The model to train.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            Return:
                :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        # ---------- call the inner loop of MAML --------------
        if self.args.run_hnet_in_batch_mode_per_task:
            if self.use_amp:
                with autocast():
                    loss = self.hnet_inner_step_per_task_batch_model(inputs, model=model)
            else:
                loss = self.hnet_inner_step_per_task_batch_model(inputs, model=model)
        else:
            if self.use_amp:
                with autocast():
                    loss = self.hnet_inner_step(inputs, model=model)
            else:
                loss = self.hnet_inner_step(inputs, model=model)
            
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        # ----------- update_steps -----------
        self.train_steps += 1

        return loss.detach()

    def hnet_inner_step_per_task_batch_model(self, inputs=None, model=None):
        """
            Run a forward loop with input/output pairs using the main language model and compute the gradients. 
            Compute the hyperparameters using gradient and instructions. 
            Update the main_lm parameters with the hyperparameters.
            NOTE: We should have the main_model frozen while updating the hnet. Then have the hnet frozen and update the main model.
            Constrain updates to to KL divergence in the main vs updated param, is this sneccesary?

            NOTE: Should we have joint update or individual update? 
            NOTE:Should we have a some warm up time for the hnet (since some its parameters are starting from random)
        """
        task_list = list(inputs.keys())
        output_loss = {"loss":0.0}
        losses = []
        for index, task in enumerate(task_list):
            input_task = inputs[task]
            if self.args.hnet_opt_mode == "hnet":
                # update the hnet model, with the main model kept frozen
                outputs = model(**input_task, forward_mode="hnet", batch_with_same_task=self.args.run_hnet_in_batch_mode_per_task)
            
            if self.args.hnet_opt_mode == "alternating_hnet_main":
                """ 
                    Alternately update the hnet model and main lm in forward modes.
                    The difference with the separate and joint is that only one mode per batch is used, which means that the hnet/main lm would have been updated in the prior bacjward step before using the other one.  
                    While in the other modes, both modes are used in a single forward batch. 
                """
                outputs = model(**input_task, forward_mode=self.toggle_train_forward_mode, batch_with_same_task=self.args.run_hnet_in_batch_mode_per_task)
                """ 
                    Toggle forward mode every gradient accumulation steps. 
                    Thus the HNET or the LM parameters would have been updated the next time the forward happens 
                """
                if (self.train_steps + 1) % (self.args.gradient_accumulation_steps * self.args.hnet_alternating_num_steps) == 0:
                    if self.toggle_train_forward_mode == "hnet":
                        self.toggle_train_forward_mode = "main_lm"
                    else:
                        self.toggle_train_forward_mode = "hnet"

            if self.args.hnet_opt_mode == "separate_hnet_main":
                    """
                        update both hnet and main model at the same time, with gradients computed by keeping the other frozen. The difference with the alternating mode, is that here the main and hnet model have not been updated through backward step prior to using each in the different modes
                    """
                    outputs_hnet = model(**input_task, forward_mode="hnet", batch_with_same_task=self.args.run_hnet_in_batch_mode_per_task)
                    outputs_main_lm = model(**input_task, forward_mode="main_lm")
                    loss = (outputs_hnet["loss"] + outputs_main_lm["loss"]) / 2.0
                    outputs = {"loss":loss}

            if self.args.hnet_opt_mode == "joint_hnet_main":
                    #  update both hnet and main model at the same time, with gradients computed for each.
                    outputs = model(**input_task, forward_mode="joint_hnet_main", batch_with_same_task=self.args.run_hnet_in_batch_mode_per_task)

            losses.append(outputs["loss"])
        # ------------ Compute loss across tasks and return to outer loop ------------
        output_loss["loss"] = torch.mean(torch.vstack(losses))
        return self.compute_loss(inputs, output_loss, return_outputs=False)

    def hnet_inner_step(self, inputs=None, model=None):
        """
            Run a forward loop with input/output pairs using the main language model and compute the gradients. 
            Compute the hyperparameters using gradient and instructions. 
            Update the main_lm parameters with the hyperparameters.
            NOTE: We should have the main_model frozen while updating the hnet. Then have the hnet frozen and update the main model.
            Constrain updates to to KL divergence in the main vs updated param, is this sneccesary?

            NOTE: Should we have joint update or individual update? 
            NOTE:Should we have a some warm up time for the hnet (since some its parameters are starting from random)
        """
        
        if self.args.hnet_opt_mode == "hnet":
            # update the hnet model, with the main model kept frozen
            outputs = model(**inputs, forward_mode="hnet")
        
        if self.args.hnet_opt_mode == "alternating_hnet_main":
            """ 
                Alternately update the hnet model and main lm in forward modes.
                The difference with the separate and joint is that only one mode per batch is used, which means that the hnet/main lm would have been updated in the prior bacjward step before using the other one.  
                While in the other modes, both modes are used in a single forward batch. 
            """
            outputs = model(**inputs, forward_mode=self.toggle_train_forward_mode)
            """ 
                Toggle forward mode every gradient accumulation steps. 
                Thus the HNET or the LM parameters would have been updated the next time the forward happens 
            """
            if (self.train_steps + 1) % (self.args.gradient_accumulation_steps * self.args.hnet_alternating_num_steps) == 0:
                if self.toggle_train_forward_mode == "hnet":
                    self.toggle_train_forward_mode = "main_lm"
                else:
                    self.toggle_train_forward_mode = "hnet"

        if self.args.hnet_opt_mode == "separate_hnet_main":
                """
                    update both hnet and main model at the same time, with gradients computed by keeping the other frozen. The difference with the alternating mode, is that here the main and hnet model have not been updated through backward step prior to using each in the different modes
                """
                outputs_hnet = model(**inputs, forward_mode="hnet")
                outputs_main_lm = model(**inputs, forward_mode="main_lm")
                loss = (outputs_hnet["loss"] + outputs_main_lm["loss"]) / 2.0
                outputs = {"loss":loss}

        if self.args.hnet_opt_mode == "joint_hnet_main":
                #  update both hnet and main model at the same time, with gradients computed for each.
                outputs = model(**inputs, forward_mode="joint_hnet_main")
        return self.compute_loss(inputs, outputs, return_outputs=False)
        
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None,) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Perform an evaluation step on :obj:`model` using obj:`inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (:obj:`nn.Module`):
                    The model to evaluate.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (:obj:`bool`):
                    Whether or not to return the loss only.

            Return:
                Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
                labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step( model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        """
            Need to make sure that decoder_input_ids is set in get_kwargs
            This should be available in the inputs
        """
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        generated_tokens = self.model.generate(**inputs, **gen_kwargs,)
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def compute_loss(self, inputs, outputs, return_outputs=False):
        """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:# We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class TrainerClassHNETMAML(TrainerClassHNET):

    def _prepare_inputs_for_train(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        We override the standard one as this one is nested and needs to be kept nested for further processing in the MAML
            Inputs:
                - inputs: dict of dicts,
                    - with task_id as key
                    - has task_batch_size_per_iter # of tasks
                - inputs[task_id] is a dict
                    -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                - inputs[task_id][key] is a tensor of length task_batch_size_per_iter
            Returns:
                input_dict: a dict wrapping the nested dict to make sure the inputs to the forward is general
                    - input_dict["inputs"]: dict of dicts,
                        - with task_id as key
                        - has task_batch_size_per_iter # of tasks
                    - input_dict["inputs"][task_id] is a dict
                        -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                    - input_dict["inputs"][task_id][key] is a tensor of length task_batch_size_per_iter
            """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        # -------- wrap the return dict within another dict to keep forward args to a single param -----------
        input_dict = {}
        input_dict["inputs"] = inputs
        return input_dict

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
            Perform a training step on a batch of inputs.
            This is essentially running the outer loop of MAML while the model.forward is running the inner loop of MAML
            Args:
                model (:obj:`nn.Module`):
                    The model to train.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            Return:
                :obj:`torch.Tensor`: The tensor with training loss on this batch.
            """
        model.train()
        inputs = self._prepare_inputs_for_train(inputs)
        # ---------- call the inner loop of MAML --------------
        if self.use_amp:
            with autocast():
                loss = self.maml_hnet_inner_loop(**inputs, model=model)
        else:
            # loss = self.maml_inner_loop(**inputs, model=model)
            loss = self.maml_hnet_inner_loop(**inputs, model=model)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()

    def hnet_inner_step(self, inputs=None, model=None):
        """
            Run a forward loop with input/output pairs using the main language model and compute the gradients. 
            Compute the hyperparameters using gradient and instructions. 
            Update the main_lm parameters with the hyperparameters.
            NOTE: We should have the main_model frozen while updating the hnet. Then have the hnet frozen and update the main model.
            Constrain updates to to KL divergence in the main vs updated param, is this sneccesary?

            NOTE: Should we have joint update or individual update? 
            NOTE:Should we have a some warm up time for the hnet (since some its parameters are starting from random)
        """
        
        if self.args.hnet_opt_mode == "hnet":
            # update the hnet model, with the main model kept frozen
            outputs = model(**inputs, forward_mode="hnet", batch_w1th_same_task=True)
        
        if self.args.hnet_opt_mode == "alternating_hnet_main":
            """ 
                Alternately update the hnet model and main lm in forward modes.
                The difference with the separate and joint is that only one mode per batch is used, which means that the hnet/main lm would have been updated in the prior bacjward step before using the other one.  
                While in the other modes, both modes are used in a single forward batch. 
            """
            outputs = model(**inputs, forward_mode=self.toggle_train_forward_mode, batch_with_same_task=self.args.run_hnet_in_batch_mode_per_task)
            """ 
                Toggle forward mode every gradient accumulation steps. 
                Thus the HNET or the LM parameters would have been updated the next time the forward happens
                For hnet maml, the maml step has  model.training_arguments.num_inner_training_steps_per_iter + 1 steps, so need to toggle based on that
            """
            # NOTE: This only works if use_multi_step_loss_optimization is False
            total_maml_inner_steps = model.module.data_args.num_tasks_per_iter / 2  * (self.args.num_inner_training_steps_per_iter + 1)
            
            if (self.train_steps + 1) % (self.args.gradient_accumulation_steps * self.args.hnet_alternating_num_steps * total_maml_inner_steps) == 0:
                if self.toggle_train_forward_mode == "hnet":
                    self.toggle_train_forward_mode = "main_lm"
                else:            
                    self.toggle_train_forward_mode = "hnet"

        if self.args.hnet_opt_mode == "separate_hnet_main":
                """
                    update both hnet and main model at the same time, with gradients computed by keeping the other frozen. The difference with the alternating mode, is that here the main and hnet model have not been updated through backward step prior to using each in the different modes
                """
                outputs_hnet = model(**inputs, forward_mode="hnet")
                outputs_main_lm = model(**inputs, forward_mode="main_lm")
                loss = (outputs_hnet["loss"] + outputs_main_lm["loss"]) / 2.0
                outputs = {"loss":loss}

        if self.args.hnet_opt_mode == "joint_hnet_main":
                #  update both hnet and main model at the same time, with gradients computed for each.
                outputs = model(**inputs, forward_mode="joint_hnet_main", batch_with_same_task=self.args.run_hnet_in_batch_mode_per_task)

        return self.compute_loss(inputs, outputs, return_outputs=False)

    def maml_hnet_inner_loop(self, inputs=None, model=None):
        """
            Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
            The outer loop is run through the Trainer class, while the forward runs an inner loop
            Arguments:
                - inputs: dict of dicts,
                    - with task_id as key
                    - has task_batch_size_per_iter # of tasks
                        - inputs[task_id] is a dict
                            -  with keys = 'input_ids', 'attention_mask', 'labels', 'task_index', 'decoder_input_ids'
                            - inputs[task_id][key] is a tensor of length task_batch_size_per_iter
                - global_step
                - epoch
            Returns:
                losses
                per_task_target_preds
        """
        total_losses = []
        task_list = list(inputs.keys())
        assert len(task_list) % 2 == 0, "%s please set --num_tasks_per_iter as a multiple of 2"%task_list

        support_tasks = task_list[0: int(len(task_list)/2)]
        target_tasks = task_list[int(len(task_list)/2):]
        steps = 0
        for task_index in range(len(support_tasks)):
            task_losses = []
            if self.args.use_multi_step_loss_optimization:
                per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            # ---------- create a copy of main model and optimizer for inner loop -----------
            meta_learner = model.clone()
            if self.args.n_gpu > 1:
                meta_learner = nn.DataParallel(meta_learner)
            meta_learner = meta_learner.train()
            # --------- split inputs to support and target sets ------------
            input_dict_support = inputs[support_tasks[task_index]]
            input_dict_target = inputs[target_tasks[task_index]]
            # ----------- inner loop steps ------------
            for num_step in range(self.args.num_inner_training_steps_per_iter):
                # --------- compute loss and update params on support set ---------
                support_loss= self.hnet_inner_step(inputs=input_dict_support, model=meta_learner)
                self.train_steps += 1
                steps += 1
                if self.args.n_gpu > 1:
                    meta_learner.module.adapt(support_loss)
                else:
                    meta_learner.adapt(support_loss)
                # ---------- compute loss on target set ------------
                if self.args.use_multi_step_loss_optimization and self.state.global_step < self.args.multi_step_loss_num_steps:
                    support_loss= self.hnet_inner_step(inputs=input_dict_target, model=meta_learner)
                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                    steps += 1
                    self.train_steps += 1
                else:
                    if num_step == (self.args.num_inner_training_steps_per_iter - 1):
                        target_loss = self.hnet_inner_step(inputs=input_dict_target, model=meta_learner)
                        task_losses.append(target_loss)
                        self.train_steps += 1
                        steps += 1
            # -------------- End of inner steps. Compute losses ----------------
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            del meta_learner
        # ------------ Compute loss across tasks and return to outer loop ------------
        loss = torch.mean(torch.stack(total_losses))
        return loss
