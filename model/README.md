---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:120312
- loss:BinaryCrossEntropyLoss
base_model: dmis-lab/biobert-base-cased-v1.1
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on dmis-lab/biobert-base-cased-v1.1

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) <!-- at revision 924f12e0c3db7f156a765ad53fb6b11e7afedbc8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Describe the Enhancing NeuroImaging Genetics through Meta-Analysis (ENIGMA) Consortium', 'This review summarizes the last decade of work by the ENIGMA (Enhancing NeuroImaging Genetics through Meta Analysis) Consortium, a global alliance of over 1400 scientists across 43 countries, studying the human brain in health and disease.'],
    ['What is the mechanism of action of verubecestat?', 'After giving an update on the development and current status of new AD therapeutics, this review will focus on BACE inhibitors and, in particular, will discuss the prospects of verubecestat (MK-8931), which has reached phase III clinical trials.'],
    ['Which anticancer drugs target human topoisomerase II?', 'The stabilization of cleavage intermediates by intercalators may have a common mechanism for DNA topoisomerase I and DNA topoisomerase II.'],
    ['Is modified vaccinia Ankara effective for smallpox?', 'INTRODUCTION: To guide the use of modified vaccinia Ankara (MVA) vaccine in response to a release of smallpox virus, the immunogenicity and safety of shorter vaccination intervals, and administration by jet injector (JI), were compared to the standard schedule of administration on Days 1 and 29 by syringe and needle (S&N).'],
    ['Which fusion protein is involved in the development of Ewing sarcoma?', 'This case provides rationale for adding a Ewing sarcoma arm to SARC024, a phase II study of regorafenib, another multi-targeted kinase inhibitor, in patients with liposarcoma, osteosarcoma and Ewing and Ewing-like sarcomas (NCT02048371).'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Describe the Enhancing NeuroImaging Genetics through Meta-Analysis (ENIGMA) Consortium',
    [
        'This review summarizes the last decade of work by the ENIGMA (Enhancing NeuroImaging Genetics through Meta Analysis) Consortium, a global alliance of over 1400 scientists across 43 countries, studying the human brain in health and disease.',
        'After giving an update on the development and current status of new AD therapeutics, this review will focus on BACE inhibitors and, in particular, will discuss the prospects of verubecestat (MK-8931), which has reached phase III clinical trials.',
        'The stabilization of cleavage intermediates by intercalators may have a common mechanism for DNA topoisomerase I and DNA topoisomerase II.',
        'INTRODUCTION: To guide the use of modified vaccinia Ankara (MVA) vaccine in response to a release of smallpox virus, the immunogenicity and safety of shorter vaccination intervals, and administration by jet injector (JI), were compared to the standard schedule of administration on Days 1 and 29 by syringe and needle (S&N).',
        'This case provides rationale for adding a Ewing sarcoma arm to SARC024, a phase II study of regorafenib, another multi-targeted kinase inhibitor, in patients with liposarcoma, osteosarcoma and Ewing and Ewing-like sarcomas (NCT02048371).',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 120,312 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                        | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                            | float                                                          |
  | details | <ul><li>min: 14 characters</li><li>mean: 59.26 characters</li><li>max: 215 characters</li></ul> | <ul><li>min: 12 characters</li><li>mean: 179.42 characters</li><li>max: 1494 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.49</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                          | sentence_1                                                                                                                                                                                                                                                         | label            |
  |:----------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Describe the Enhancing NeuroImaging Genetics through Meta-Analysis (ENIGMA) Consortium</code> | <code>This review summarizes the last decade of work by the ENIGMA (Enhancing NeuroImaging Genetics through Meta Analysis) Consortium, a global alliance of over 1400 scientists across 43 countries, studying the human brain in health and disease.</code>       | <code>1.0</code> |
  | <code>What is the mechanism of action of verubecestat?</code>                                       | <code>After giving an update on the development and current status of new AD therapeutics, this review will focus on BACE inhibitors and, in particular, will discuss the prospects of verubecestat (MK-8931), which has reached phase III clinical trials.</code> | <code>1.0</code> |
  | <code>Which anticancer drugs target human topoisomerase II?</code>                                  | <code>The stabilization of cleavage intermediates by intercalators may have a common mechanism for DNA topoisomerase I and DNA topoisomerase II.</code>                                                                                                            | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0665 | 500   | 0.5656        |
| 0.1330 | 1000  | 0.5177        |
| 0.1995 | 1500  | 0.4975        |
| 0.2660 | 2000  | 0.4939        |
| 0.3324 | 2500  | 0.4884        |
| 0.3989 | 3000  | 0.4801        |
| 0.4654 | 3500  | 0.4823        |
| 0.5319 | 4000  | 0.4713        |
| 0.5984 | 4500  | 0.4633        |
| 0.6649 | 5000  | 0.4653        |
| 0.7314 | 5500  | 0.4659        |
| 0.7979 | 6000  | 0.4638        |
| 0.8644 | 6500  | 0.4679        |
| 0.9309 | 7000  | 0.4513        |
| 0.9973 | 7500  | 0.4488        |
| 1.0638 | 8000  | 0.4157        |
| 1.1303 | 8500  | 0.4124        |
| 1.1968 | 9000  | 0.4053        |
| 1.2633 | 9500  | 0.4096        |
| 1.3298 | 10000 | 0.4112        |
| 1.3963 | 10500 | 0.4115        |
| 1.4628 | 11000 | 0.3958        |
| 1.5293 | 11500 | 0.4019        |
| 1.5957 | 12000 | 0.3996        |
| 1.6622 | 12500 | 0.3996        |
| 1.7287 | 13000 | 0.3978        |
| 1.7952 | 13500 | 0.3987        |
| 1.8617 | 14000 | 0.3937        |
| 1.9282 | 14500 | 0.3967        |
| 1.9947 | 15000 | 0.3863        |
| 0.0665 | 500   | 0.3881        |
| 0.1330 | 1000  | 0.401         |
| 0.1995 | 1500  | 0.4115        |
| 0.2660 | 2000  | 0.414         |
| 0.3324 | 2500  | 0.4077        |
| 0.3989 | 3000  | 0.411         |
| 0.4654 | 3500  | 0.4096        |
| 0.5319 | 4000  | 0.4103        |
| 0.5984 | 4500  | 0.417         |
| 0.6649 | 5000  | 0.4026        |
| 0.7314 | 5500  | 0.416         |
| 0.7979 | 6000  | 0.4121        |
| 0.8644 | 6500  | 0.411         |
| 0.9309 | 7000  | 0.4033        |
| 0.9973 | 7500  | 0.4082        |
| 1.0638 | 8000  | 0.3704        |
| 1.1303 | 8500  | 0.3615        |
| 1.1968 | 9000  | 0.353         |
| 1.2633 | 9500  | 0.3637        |
| 1.3298 | 10000 | 0.3631        |
| 1.3963 | 10500 | 0.3661        |
| 1.4628 | 11000 | 0.3633        |
| 1.5293 | 11500 | 0.3597        |
| 1.5957 | 12000 | 0.361         |
| 1.6622 | 12500 | 0.3599        |
| 1.7287 | 13000 | 0.3638        |
| 1.7952 | 13500 | 0.372         |
| 1.8617 | 14000 | 0.3564        |
| 1.9282 | 14500 | 0.3601        |
| 1.9947 | 15000 | 0.3614        |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.2
- PyTorch: 2.9.0+cu126
- Accelerate: 1.12.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->