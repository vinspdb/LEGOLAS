import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


class LongFormerMultiClassificationHeads(nn.Module):
    def __init__(self, longformer):
        super(LongFormerMultiClassificationHeads, self).__init__()
        self.longformer = longformer
        self.output_layer = nn.Linear(longformer.config.hidden_size, 8)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.output_layer(outputs.pooler_output)
        return pooled_output


class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, max_seq_len: int, gpt_model):
        super(SimpleGPT2SequenceClassifier, self).__init__()
        self.gpt2model = gpt_model
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)

    def forward(self, input_id, mask):
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output
