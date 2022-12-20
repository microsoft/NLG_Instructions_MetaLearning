"""Code for defining the various models for abstractive rewrite.

The current code is tested with base seq2seq models, as well as distilled models specific to summarization.
Some of these are as follows:
    "t5-small", "t5-base", "facebook/bart-base", ""sshleifer/distilbart-xsum-6-6", ""sshleifer/distilbart-cnn-12-3",
    "patrickvonplaten/bert2bert_cnn_daily_mail" and Bert2Bert using minilm-bert-l6-h384-uncased.
"""
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoConfig, AutoTokenizer

import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

import os
import pdb

CONFIG_NAME = "config.json"
DATA_ARGS_NAME = "data_args.bin"
MODEL_ARGS_NAME = "model_args.bin"
TRAINING_ARGS_NAME = "model_training_args.bin"
WEIGHTS_NAME = "pytorch_model.bin"

class BaseLanguageModel(nn.Module):
    def __init__(self, model_args, training_args, data_args, logger):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.logger = logger

        # load base model and tokenizer
        self.create_config()
        self.load_tokenizer()
        self.load_main_lm()

    def to(self, device):
        self.device = device
        super().to(device)

    def create_config(self):
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        if self.training_args.gradient_checkpointing:
            self.config.gradient_checkpointing = self.training_args.gradient_checkpointing
            self.config.use_cache = False

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

    def load_main_lm(self):
        return

    def load_initial_checkpoint(self):
        """Function to load pre-trained or fine-tuned weights from the file 'python_model.bin' in the model directory
        Internal parameters used by this function:
        model_name_or_path : An attribute of self.params
        main_lm: The PreTrainedModel instance which was initialzed in the constructor of derived class.
        """
        if torch.cuda.is_available():
            ckpt = torch.load(os.path.join(self.params.model_name_or_path, WEIGHTS_NAME))
        else:
            ckpt = torch.load(os.path.join(self.params.model_name_or_path, WEIGHTS_NAME), map_location=torch.device("cpu"))
        try:
            self.main_lm.load_state_dict(ckpt)
        except Exception as e:
            print(e)

    def prepare_decoder_input_ids_from_labels(self, labels=None):
        # this is used in the collator to find the inputs so adding it here
        return self.main_lm.prepare_decoder_input_ids_from_labels(labels=labels)

    def freeze_main_lm(self):
        for param in self.main_lm.parameters():
            param.requires_grad = False

    def save_extra_args(self, output_dir=None):
        """Saves model and data arguments to restore model from directory.

        To be used when initial config is unknown, for example, during inference.
        Note: Training arguments are not the same as the one saved by trainer class.
        """
        torch.save(self.model_args, os.path.join(output_dir, MODEL_ARGS_NAME))
        torch.save(self.data_args, os.path.join(output_dir, DATA_ARGS_NAME))
        torch.save(self.training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    @classmethod
    def from_pretrained(cls, model_dirpath, logger):
        model_args = torch.load(os.path.join(model_dirpath, MODEL_ARGS_NAME))
        data_args = torch.load(os.path.join(model_dirpath, DATA_ARGS_NAME))
        training_args = torch.load(os.path.join(model_dirpath, TRAINING_ARGS_NAME))
        return cls(model_args, training_args, data_args, logger)

class BaseS2SModel(BaseLanguageModel):
    """
        A container class for Sequence to Sequence models such as BART
    """
    def __init__(self, model_args, training_args, data_args, logger):
        super().__init__(model_args, training_args, data_args, logger)

    def load_main_lm(self):
        self.main_lm = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path, from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config, cache_dir=self.model_args.cache_dir, revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None)
        self.main_lm.resize_token_embeddings(len(self.tokenizer))

        if self.main_lm.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.training_args.label_smoothing_factor > 0 and not hasattr(self.main_lm, "prepare_decoder_input_ids_from_labels"):
            self.logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self.main_lm.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

    def generate(self, input_ids, attention_mask=None, **gen_kwargs,):
        """
            To generate with ouput prefix, we should encode it in the decoder_input_ids.
            Otherwise it uses a default start token as the decoder input IDs in /mnt/Repos/transformers/src/transformers/generation_utils.py (928)
        """
        max_length = self.model_args.decode_max_length + gen_kwargs['decoder_input_ids'].shape[1]
        max_length =  min(max_length, self.data_args.max_target_length)
        gen_outputs = self.main_lm.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,
            early_stopping=True, num_beams=self.model_args.decode_num_beams,
            length_penalty=self.model_args.decode_length_penalty, no_repeat_ngram_size=3,
            decoder_input_ids=gen_kwargs['decoder_input_ids'])
        return gen_outputs

    def forward(
        self,
        attention_mask=None,
        cross_attn_head_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        head_mask=None,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        return_dict=None,
        use_cache=None,
        task_index=None,
        **gen_kwargs,
        ):
            """
                returns: Seq2SeqLMOutput(
                    loss=masked_lm_loss,
                    logits=lm_logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,)
            """
            output =  self.main_lm(
                input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels, use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,)

            return output

class BaseCLMModel(BaseLanguageModel):
    """
        A container class for AutoRegressive/Causal Language Models such as GPT2
    """
    def __init__(self, model_args, training_args, data_args, logger):
        super().__init__(model_args, training_args, data_args, logger)
    
    def load_main_lm(self):
        self.main_lm = AutoModelForCausalLM.from_pretrained(self.model_args.model_name_or_path, from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
        config=self.config, cache_dir=self.model_args.cache_dir, revision=self.model_args.model_revision, use_auth_token=True if self.model_args.use_auth_token else None,)
        
        print('adapting the size of the model embedding to include [PAD]')
        print('len(tokenizer) = ', len(self.tokenizer))
        num_added_tokens = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        embedding_layer = self.main_lm.resize_token_embeddings(len(self.tokenizer))
        print('len(tokenizer) = ', len(self.tokenizer))
        print(self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        print(self.tokenizer.bos_token, self.tokenizer.bos_token_id) 
    
    def generate(self, input_ids, attention_mask=None, **gen_kwargs,):
        gen_outputs = self.main_lm.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=self.data_args.max_target_length,
            early_stopping=True, num_beams=self.model_args.decode_num_beams,
            length_penalty=self.model_args.decode_length_penalty, no_repeat_ngram_size=3,
            decoder_input_ids=gen_kwargs['decoder_input_ids'])

        return gen_outputs
    
    def forward(
        self,
        attention_mask=None,
        head_mask=None,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        position_ids=None,
        return_dict=None,
        token_type_ids=None,
        use_cache=None,
        decoder_input_ids=None
        ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``

            Returns: CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,)
        """
        return  self.main_lm(input_ids=input_ids, labels=labels, attention_mask=attention_mask, head_mask=head_mask,  inputs_embeds=inputs_embeds, output_attentions=output_attentions,output_hidden_states=output_hidden_states, past_key_values=past_key_values, position_ids=position_ids, return_dict=return_dict, token_type_ids=token_type_ids, use_cache=use_cache)

    def compute_loss(self, lm_logits, labels):
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        bsz, seqlen, vocab_size = shift_logits.shape
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bsz, seqlen).sum(dim=-1)
        loss =  loss.mean()
        return loss
