"""
        Adapted from KnowledgeEditor paper: https://github.com/nicola-decao/KnowledgeEditor/tree/main/src/models
"""

from xml.etree.ElementInclude import include
from .models_pt import BaseS2SModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration
from transformers.trainer_pt_utils import nested_concat
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random

from higher.patch import monkeypatch as make_functional
import pdb

class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):

        # if len(self.parameter_shape) == 2:
        #     split_dims = [self.parameter_shape[1], self.parameter_shape[0], self.parameter_shape[1], self.parameter_shape[0], 1,]
        #     alpha, beta, gamma, delta, eta = self.conditioners(inputs).split(split_dims, dim=-1,)
        #     a = beta.softmax(-1).T @ alpha
        #     b = delta.softmax(-1).T @ gamma
        # elif len(self.parameter_shape) == 1:
        #     split_dims = [self.parameter_shape[0], self.parameter_shape[0], 1]
        #     a, b, eta = self.conditioners(inputs).split(split_dims, dim=-1)
        # else:
        #     raise RuntimeError()
        
        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a_cond = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b_cond = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a_cond, b_cond, conditioner_norm = self.conditioners(inputs).split([self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1)
        else:
            raise RuntimeError()

        # ------------ Set grad to 1.0 if it is None ----------
        if grad is None:
            grad = 1.0
        return (self.max_scale * conditioner_norm.sigmoid().squeeze() * (grad * a_cond.squeeze() + b_cond.squeeze()))

class BartForConditionalGenerationDerived(BartForConditionalGeneration):

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = self.shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.last_hidden_state, # NOTE: changed this from the original output to get the last hidden states
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class HNET_Conditioner(torch.nn.Module):
    def __init__(self, model_args, training_args, data_args, logger, model, model_params_to_include={}):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.model = model
        self.hnet_condition_dim = [p for n, p in model.named_parameters()][-1].shape[-1]
        self.create_config()
        self.load_tokenizer()
        self.load_hnet_lm()

        if "restricted" in self.model_args.hnet_exclude_encoder_or_decoder:
            self.excluded_layers = ["bias", "norm", "embeddings", "classifier", "pooler", "shared", "embed", "positions",]
        else:
            self.excluded_layers = []

        if self.model_args.hnet_include_layer is not "none":
            if self.model_args.hnet_include_layer in self.excluded_layers:
                self.excluded_layers.remove(self.model_args.hnet_include_layer)

        
        if "none" not in  self.model_args.hnet_exclude_encoder_or_decoder:
            if "encoder" in self.model_args.hnet_exclude_encoder_or_decoder:
                self.excluded_layers.append("model.encoder")
            if "decoder" in self.model_args.hnet_exclude_encoder_or_decoder:
                self.excluded_layers.append("model.decoder")
        # NOTE: get the list later from model parameters
        self.model_params_to_include = [n for n, _ in self.model.named_parameters()
                                        if all(e not in n.lower() for e in self.excluded_layers) ]
        self.create_param_map_for_main_lm(model)
    
    def load_hnet_lm(self):
        # self.hnet_lm = AutoModelForSeq2SeqLM.from_pretrained(
        #     self.model_args.hnet_model_name_or_path, from_tf=bool(".ckpt" in self.model_args.hnet_model_name_or_path),
        #     config=self.hnet_config, cache_dir=self.model_args.cache_dir, revision=self.model_args.model_revision,
        #     use_auth_token=True if self.model_args.use_auth_token else None)

        self.hnet_lm = BartForConditionalGenerationDerived.from_pretrained(
            self.model_args.hnet_model_name_or_path, from_tf=bool(".ckpt" in self.model_args.hnet_model_name_or_path),
            config=self.hnet_config, cache_dir=self.model_args.cache_dir, revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None)

        self.hnet_lm.resize_token_embeddings(len(self.hnet_tokenizer))

        if self.hnet_lm.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.training_args.label_smoothing_factor > 0 and not hasattr(self.main_lm, "prepare_decoder_input_ids_from_labels"):
            self.logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self.hnet_lm.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )
    
    def create_config(self):
        #  for now let us assume that the main model and the hnet model are the same type to reduce the complexity of different types of tokenization
        self.hnet_config = AutoConfig.from_pretrained(
            self.model_args.hnet_config_name if self.model_args.hnet_config_name else self.model_args.hnet_model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        if self.training_args.gradient_checkpointing:
            self.hnet_config.gradient_checkpointing = self.training_args.gradient_checkpointing
            self.hnet_config.use_cache = False

    def load_tokenizer(self):
        self.hnet_tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.hnet_tokenizer_name if self.model_args.hnet_tokenizer_name else self.model_args.hnet_model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
    
    def create_param_map_for_main_lm(self, model):
        self.param2conditioner_map = {n: "{}_conditioner".format(n).replace(".", "_") for n, p in model.named_parameters() 
                                                                                        if n in self.model_params_to_include}

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    self.hnet_condition_dim,
                    self.model_args.hnet_hidden_dim,
                    max_scale=self.model_args.hnet_max_scale,
                )
                for n, p in self.model.named_parameters() if n in self.model_params_to_include
            }
        )

        # not much difference between 1 and 2
        # 2: self.decoder_input_ids_hnet = torch.tensor([list(range(len(self.conditioners)))])
        # self.decoder_input_ids_hnet = None # 3
        self.decoder_input_ids_hnet = torch.tensor([[2] * len(self.conditioners)])

    def forward(self, input_ids_hnet=None, attention_mask_hnet = None, decoder_input_ids_hnet = None, grads = None):
        """
            returns: Params Dict of the main model
        """
        
        if self.model_args.hnet_use_encoder_last_hidden_state:
            """
                Here we take the last hidden state of the encoder and then project it for conditioning.
            """
            output_hnet_lm =  self.hnet_lm(input_ids=input_ids_hnet, attention_mask=attention_mask_hnet, decoder_input_ids=decoder_input_ids_hnet)
            output_hnet_lm =  output_hnet_lm["encoder_last_hidden_state"][:,-1,:]
            main_model_params = {p: self.conditioners[self.param2conditioner_map[p]](output_hnet_lm, grad=grads[p] if grads else None,) for p, c in self.param2conditioner_map.items()}
        else:
            """
                Since last hidden state may not contain enough information for all the parameters separately, we use the hidden state of the decoder and use the first k tokens for projecting (where k is the size of the param set), and thus each index represents a separate param for the main lm and thus may be able to use the full bandwidth of the HNET model
            """
            if self.decoder_input_ids_hnet is not None:
                output_hnet_lm =  self.hnet_lm(input_ids=input_ids_hnet, attention_mask=attention_mask_hnet, decoder_input_ids=self.decoder_input_ids_hnet.to(input_ids_hnet.device))
            else:
                output_hnet_lm =  self.hnet_lm(input_ids=input_ids_hnet, attention_mask=attention_mask_hnet, decoder_input_ids=None)

            main_model_params = {}
            for index, map in enumerate(self.param2conditioner_map.items()):
                p, c = map
                grad = grads[p] if grads else None
                output_hnet =  output_hnet_lm["decoder_hidden_states"][:,index,:]
                main_model_params[p] = self.conditioners[self.param2conditioner_map[p]](output_hnet, grad=grad)

        return main_model_params

class S2SHNETModel(BaseS2SModel):
    def __init__(self, model_args, training_args, data_args, logger):
        super().__init__(model_args, training_args, data_args, logger)
        self.hnet_model = HNET_Conditioner(model_args, training_args, data_args, logger, self.main_lm)
        if self.model_args.hnet_weighet_delta_params:
            self.alpha = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def get_logits_grads_from_orig_params_dict(self,  
                                                attention_mask=None,
                                                decoder_attention_mask=None,
                                                decoder_input_ids=None,
                                                input_ids=None,):
        
        return None, None
        # TODO: implement the grads etc. For now we do not use the grads
        with torch.enable_grad():
            logits_orig, logit_for_grad, _ = self.main_lm.eval()(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
            ).logits.split(
                [
                    len(batch["src_input_ids"]) - (2 if self.hparams.use_views else 1),
                    1,
                    1 if self.hparams.use_views else 0,
                ]
            )

            logits_orig = logits_orig.detach()

            grads = torch.autograd.grad(
                label_smoothed_nll_loss(
                    logit_for_grad.log_softmax(-1),
                    batch["trg_input_ids"][
                        -2
                        if self.hparams.use_views
                        else -1 : -1
                        if self.hparams.use_views
                        else None,
                        1:,
                    ],
                    epsilon=self.hparams.eps,
                    ignore_index=self.tokenizer.pad_token_id,
                )[1]
                / batch["trg_attention_mask"][
                    -2
                    if self.hparams.use_views
                    else -1 : -1
                    if self.hparams.use_views
                    else None,
                    1:,
                ].sum(),
                self.model.parameters(),
            )
            grads = {
                name: grad
                for (name, _), grad in zip(self.model.named_parameters(), grads)
            }

    def forward(
        self,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        input_ids=None,
        labels=None,
        input_ids_hnet = None,
        attention_mask_hnet = None,
        decoder_input_ids_hnet = None,
        forward_mode = "main_lm",
        batch_with_same_task=False, 
        **gen_kwargs,
        ):

        batch_size = input_ids.shape[0]
        output_loss = {"loss":0.0}
        losses = []

        #  The Hnet forward works in a single batch so need to iterate over multiple items in the batch
        # if batch_wuth_same_task:
        # pdb.set_trace()
        if batch_with_same_task:
            output = self.forward_batch(
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                input_ids=input_ids,
                labels=labels,
                input_ids_hnet = input_ids_hnet,
                attention_mask_hnet = attention_mask_hnet,
                forward_mode = forward_mode,
                **gen_kwargs,)
            output_loss["loss"] = output["loss"]
        else:       
            for index in range(batch_size):
                output = self.forward_single_item(
                    attention_mask=attention_mask[index].unsqueeze(0),
                    decoder_input_ids=decoder_input_ids[index].unsqueeze(0),
                    input_ids=input_ids[index].unsqueeze(0),
                    labels=labels[index].unsqueeze(0),
                    input_ids_hnet = input_ids_hnet[index].unsqueeze(0),
                    attention_mask_hnet = attention_mask_hnet[index].unsqueeze(0),
                    forward_mode = forward_mode,
                    **gen_kwargs,)
                losses.append(output["loss"])
            output_loss["loss"] = torch.mean(torch.vstack(losses))
        return output_loss

    def forward_batch(
        self,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        input_ids=None,
        labels=None,
        input_ids_hnet = None,
        attention_mask_hnet = None,
        decoder_input_ids_hnet = None,
        forward_mode = "main_lm",
        **gen_kwargs,
        ):   
        """
            This is for the case where the entire batch is of a single task, thus the HNET can be forwarded once
        """
        grads = None
        if forward_mode == "hnet":
            # Here we keep the main_lm frozen and just learn the hnet model
            if self.model_args.hnet_use_eval_for_nograd:
                output_hnet_params_dict =  self.hnet_model.train()(input_ids_hnet=input_ids_hnet[0].unsqueeze(0), attention_mask_hnet=attention_mask_hnet[0].unsqueeze(0), decoder_input_ids_hnet=None, grads=grads)
            else:
                for n, p in self.hnet_model.named_parameters():
                    p.requires_grad = True
                output_hnet_params_dict =  self.hnet_model.train()(input_ids_hnet=input_ids_hnet[0].unsqueeze(0), attention_mask_hnet=attention_mask_hnet[0].unsqueeze(0), decoder_input_ids_hnet=None, grads=grads)

            updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            
            if self.model_args.hnet_use_eval_for_nograd:        
                fn_main_lm =  make_functional(self.main_lm).eval()
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
            else:
                for n, p in self.main_lm.named_parameters():
                    p.requires_grad = False
                fn_main_lm =  make_functional(self.main_lm).train()
                for n, p in fn_main_lm.named_parameters():
                    p.requires_grad = False
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)

        if forward_mode == "main_lm":
            # Here we keep the hnet frozen and just learn the main model
            if self.model_args.hnet_use_eval_for_nograd:
                output_hnet_params_dict =  self.hnet_model.eval()(input_ids_hnet=input_ids_hnet[0].unsqueeze(0), attention_mask_hnet=attention_mask_hnet[0].unsqueeze(0), decoder_input_ids_hnet=None, grads=grads)
            else:
                for n, p in self.hnet_model.named_parameters():
                    p.requires_grad = False
                output_hnet_params_dict =  self.hnet_model(input_ids_hnet=input_ids_hnet[0].unsqueeze(0), attention_mask_hnet=attention_mask_hnet[0].unsqueeze(0), decoder_input_ids_hnet=None, grads=grads)

            if self.model_args.hnet_use_eval_for_nograd:        
                fn_main_lm =  make_functional(self.main_lm).train()
                updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
            else:
                for n, p in self.main_lm.named_parameters():
                    p.requires_grad = True
                fn_main_lm =  make_functional(self.main_lm).train() 
                for n, p in fn_main_lm.named_parameters():
                    p.requires_grad = True
                updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
        
        if forward_mode == "joint_hnet_main":
            # Here we learn both the hnet and the main model
            output_hnet_params_dict =  self.hnet_model.train()(input_ids_hnet=input_ids_hnet[0].unsqueeze(0), attention_mask_hnet=attention_mask_hnet[0].unsqueeze(0), decoder_input_ids_hnet=None, grads=grads)

            fn_main_lm =  make_functional(self.main_lm).train()
            updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
        return output

    def forward_single_item(
        self,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        input_ids=None,
        labels=None,
        input_ids_hnet = None,
        attention_mask_hnet = None,
        decoder_input_ids_hnet = None,
        forward_mode = "main_lm",
        **gen_kwargs,
        ):
        """
            forward mode = {"hnet", "main_lm"}
            hnet: main model is kept frozen while the hnet is updated
            main_lm: hnet is kept frozen fron while the main model is updated

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
        original_logits, grads = self.get_logits_grads_from_orig_params_dict(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        if forward_mode == "hnet":
            # Here we keep the main_lm frozen and just learn the hnet model
            if self.model_args.hnet_use_eval_for_nograd:
                output_hnet_params_dict =  self.hnet_model.train()(input_ids_hnet=input_ids_hnet, attention_mask_hnet=attention_mask_hnet, decoder_input_ids_hnet=None, grads=grads)
            else:
                for n, p in self.hnet_model.named_parameters():
                    p.requires_grad = True
                output_hnet_params_dict =  self.hnet_model.train()(input_ids_hnet=input_ids_hnet, attention_mask_hnet=attention_mask_hnet, decoder_input_ids_hnet=None, grads=grads)
            updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            
            if self.model_args.hnet_use_eval_for_nograd:        
                fn_main_lm =  make_functional(self.main_lm).eval()
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
            else:
                for n, p in self.main_lm.named_parameters():
                    p.requires_grad = False
                fn_main_lm =  make_functional(self.main_lm).train()
                for n, p in fn_main_lm.named_parameters():
                    p.requires_grad = False
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)

        if forward_mode == "main_lm":
            # Here we keep the hnet frozen and just learn the main model
            if self.model_args.hnet_use_eval_for_nograd:        
                output_hnet_params_dict =  self.hnet_model.eval()(input_ids_hnet=input_ids_hnet, attention_mask_hnet=attention_mask_hnet, decoder_input_ids_hnet=None, grads=grads)
            else:
                for n, p in self.hnet_model.named_parameters():
                    p.requires_grad = False
                output_hnet_params_dict =  self.hnet_model(input_ids_hnet=input_ids_hnet, attention_mask_hnet=attention_mask_hnet, decoder_input_ids_hnet=None, grads=grads)

            if self.model_args.hnet_use_eval_for_nograd:        
                fn_main_lm =  make_functional(self.main_lm).train()
                updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
            else:
                for n, p in self.main_lm.named_parameters():
                    p.requires_grad = True
                fn_main_lm =  make_functional(self.main_lm).train() 
                for n, p in fn_main_lm.named_parameters():
                    p.requires_grad = True
                updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
                output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)
        
        if forward_mode == "joint_hnet_main":
            # Here we learn both the hnet and the main model
            output_hnet_params_dict =  self.hnet_model.train()(input_ids_hnet=input_ids_hnet, attention_mask_hnet=attention_mask_hnet, decoder_input_ids_hnet=None, grads=grads)

            fn_main_lm =  make_functional(self.main_lm).train()
            # updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            # if self.model_args.hnet_weighet_delta_params:
            #     updated_main_lm_params = [self.alpha * output_hnet_params_dict.get(n, 0) + (1-self.alpha) * p for n, p in self.main_lm.named_parameters()]
            # else:
            updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            output =  fn_main_lm(input_ids=input_ids, attention_mask=attention_mask,decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, params=updated_main_lm_params, labels=labels)

        return output
    
    def generate(self, 
        attention_mask=None,
        decoder_input_ids=None,
        input_ids=None,
        input_ids_hnet = None,
        attention_mask_hnet = None,
        labels = None,
        **gen_kwargs,):
        """
            To generate with ouput prefix, we should encode it in the decoder_input_ids.
            Otherwise it uses a default start token as the decoder input IDs in /mnt/Repos/transformers/src/transformers/generation_utils.py (928)
            The generate does not work in data pareallel
        """
        batch_size = input_ids.shape[0]
        predictions = None
        for index in range(batch_size):
            output = self.generate_single_batch(
                attention_mask=attention_mask[index].unsqueeze(0),
                decoder_input_ids=decoder_input_ids[index].unsqueeze(0),
                input_ids=input_ids[index].unsqueeze(0),
                input_ids_hnet = input_ids_hnet[index].unsqueeze(0),
                attention_mask_hnet = attention_mask_hnet[index].unsqueeze(0),
                labels = None,
                **gen_kwargs,)
            predictions = output if predictions is None else nested_concat(predictions, output, padding_index=-100)
        return predictions
    
    def generate_single_batch(self, 
        attention_mask=None,
        decoder_input_ids=None,
        input_ids=None,
        input_ids_hnet = None,
        attention_mask_hnet = None,
        labels = None,
        **gen_kwargs,):
        with torch.no_grad():
            output_hnet_params_dict =  self.hnet_model.eval()(input_ids_hnet=input_ids_hnet, attention_mask_hnet=attention_mask_hnet, decoder_input_ids_hnet=None, grads=None)
            
            max_length = self.model_args.decode_max_length + decoder_input_ids.shape[1]
            max_length =  min(max_length, self.data_args.max_target_length)
            
            fn_main_lm =  make_functional(self.main_lm).eval()
            # updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            
            updated_main_lm_params = [output_hnet_params_dict.get(n, 0) + p for n, p in self.main_lm.named_parameters()]
            
            gen_outputs = fn_main_lm.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,
                early_stopping=True, num_beams=self.model_args.decode_num_beams,
                length_penalty=self.model_args.decode_length_penalty, no_repeat_ngram_size=3,
                decoder_input_ids=decoder_input_ids,
                params=updated_main_lm_params,)
                        
            return gen_outputs
