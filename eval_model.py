from transformers import AutoModel, AutoTokenizer
from history_dataset import TextDataset
from neural_network import LongFormerMultiClassificationHeads, SimpleGPT2SequenceClassifier
from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
import os
from pathlib import Path
from typing import Optional
from transformers import BertModel, BertConfig

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")


def get_weight_dir(
        model_ref: str,
        *,
        model_dir=HF_DEFAULT_HOME,
        revision: str = "main", ) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir

model_name = 'bertm'

if model_name == 'bertm':
    weights_dir = get_weight_dir('prajjwal1/bert-medium')
elif model_name == 'roberta':
    weights_dir = get_weight_dir('FacebookAI/roberta-base')
elif model_name == 'gpt2':
    weights_dir = get_weight_dir('openai-community/gpt2')
elif model_name == 'cbert':
    weights_dir = get_weight_dir('emilyalsentzer/Bio_ClinicalBERT')
else:
    print('model not found')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('mimic_test.pkl', 'rb') as f:
    test = pickle.load(f)

with open('mimic_label_train.pkl', 'rb') as f:
    label_train = pickle.load(f)

with open('mimic_label_test.pkl', 'rb') as f:
    label_test = pickle.load(f)

label2id = {}
id2label = {}
i = 0
for l in list(np.unique(label_train)):
    label2id[l] = i
    id2label[i] = l
    i = i + 1

label_test_int = []
for l in label_test:
    label_test_int.append(label2id[l])

if model_name =='gpt2':

    model = AutoModel.from_pretrained(weights_dir)
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=weights_dir, num_labels=8)
        # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=weights_dir, truncation_side='left')
        # default to left padding
    tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
    test_model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=8, max_seq_len=512, gpt_model=model).to(device)

else:
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
    test_model = AutoModel.from_pretrained(weights_dir)
    test_model = LongFormerMultiClassificationHeads(test_model)
    test_model.load_state_dict(torch.load(model_name+'.pth'))
    test_model = test_model.to(device)


test_dataset = TextDataset(test, label_test_int, tokenizer, 512)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


test_model.eval()

all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = test_model(input_ids, attention_mask)
        pred_prob.append(output.cpu())
        predicted = output.argmax(dim=1)
        all_targets.extend(batch['labels'].to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())

all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]


with open('prediction/'+model_name+'_prob.pkl', 'wb') as file:
    pickle.dump(pred_prob, file)

with open('prediction/'+model_name+'_all_target.pkl', 'wb') as file:
    pickle.dump(all_targets, file)

with open('prediction/'+model_name+'_all_prediction.pkl', 'wb') as file:
    pickle.dump(all_predictions, file)

report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)

result = open('prediction/mimicel_report_'+model_name+'.txt', 'w')
result.write(report)
result.write('\n')
conf_m = confusion_matrix(all_targets, all_predictions)
print(conf_m)
result.write(str(conf_m))