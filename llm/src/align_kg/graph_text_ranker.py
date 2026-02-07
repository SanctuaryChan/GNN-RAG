import torch
from torch import nn
from transformers import AutoModel


class GraphTextRanker(nn.Module):
    def __init__(self, model_name_or_path, share_encoder=True, dropout=0.1):
        super().__init__()
        self.share_encoder = share_encoder
        self.graph_encoder = AutoModel.from_pretrained(model_name_or_path)
        if share_encoder:
            self.text_encoder = self.graph_encoder
        else:
            self.text_encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden = self.graph_encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 3, 1)

    def encode(self, encoder, input_ids, attention_mask):
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return cls

    def forward(self, graph_inputs, text_inputs, labels=None, align=False, tau=0.07):
        z_g = self.encode(self.graph_encoder, **graph_inputs)
        z_t = self.encode(self.text_encoder, **text_inputs)
        z_g = self.dropout(z_g)
        z_t = self.dropout(z_t)

        feat = torch.cat([z_g, z_t, z_g * z_t], dim=-1)
        scores = self.classifier(feat).squeeze(-1)

        loss = None
        if labels is not None:
            bce = nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
            loss = bce
            if align:
                z_gn = nn.functional.normalize(z_g, p=2, dim=-1)
                z_tn = nn.functional.normalize(z_t, p=2, dim=-1)
                logits = torch.matmul(z_gn, z_tn.t()) / tau
                targets = torch.arange(logits.size(0), device=logits.device)
                align_loss = nn.functional.cross_entropy(logits, targets)
                loss = loss + align_loss
        return scores, loss
