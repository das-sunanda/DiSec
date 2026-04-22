# DiSec: Mitigating Backdoors in Pre-trained Language Models via Disentanglement of Adversarial Weights for Secure Fine-Tuning


## Environment Versions
- **Python:** 3.8.20  
- **PyTorch:** 1.10.2  
- **CUDA (PyTorch build):** 11.1  
- **Transformers:** 4.46.3  
- **Datasets:** 3.1.0  


## Installation
Install Python dependencies:
```bash
pip install -r requirements.txt
```


## Trigger Mining and Overlap Ratio Run Guide

### Script (Trigger Mining + Overlap Ratio)
```bash
python3 trigger-minning/trigger-bert.py \
  --TSV_PATH "<path_to_aux_tsv_with_sentence_column>" \
  --MODEL_DIR "<path_to_backdoored_bert_model_directory>" \
  --TOKENIZER_DIR "<path_to_tokenizer_directory_for_backdoored_model>"
```


## DiSec Defense (Detection + Correction) Run Guide

### Script (BERT)

```bash
python3 defense-algorithm/defense-bert.py \
  --auxiliary_data_path "<path_to_clean_aux_tsv_with_sentence_and_label_columns>" \
  --backdoored_model_dir "<path_to_backdoored_bert_model_directory>" \
  --out_union "<output_dir_for_union_clean_model>" \
  --out_inter "<output_dir_for_intersection_clean_model>" \
  --out_vae   "<output_dir_for_vae_only_clean_model>" \
  --out_svd   "<output_dir_for_svd_only_clean_model>" \
  --Top_K_vulnerable <top_k_layers_int> \
  --round_T <num_rounds_int> \
  --SVD_components <num_svd_components_int>
```

### Script (RoBERTa)
```bash
python3 defense-algorithm/defense-roberta.py \
  --auxiliary_data_path "<path_to_clean_aux_tsv_with_sentence_and_label_columns>" \
  --backdoored_model_dir "<path_to_backdoored_roberta_model_directory>" \
  --out_union "<output_dir_for_union_clean_model>" \
  --out_inter "<output_dir_for_intersection_clean_model>" \
  --out_vae   "<output_dir_for_vae_only_clean_model>" \
  --out_svd   "<output_dir_for_svd_only_clean_model>" \
  --Top_K_vulnerable <top_k_layers_int> \
  --exclude_layer <exclude_layer_int> \
  --round_T <num_rounds_int> \
  --SVD_components <num_svd_components_int>
```


## Fine-tunning Guide
### Script (fine-tunning)

```bash
python3 fine-tunning/BERT/sst2.py \
  --model_path "<path_to_purified_model_weights_file>" \
  --config_path "<path_to_config_json>" \
  --tokenizer_path "<path_to_tokenizer_directory>" \
  --triggers "<list_of_trigger_tokens_as_a_string>"
```
