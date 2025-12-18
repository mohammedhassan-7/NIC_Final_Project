import torch
import torch.nn as nn
from transformers import AutoModel

class TinyBERTClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.3):
        super(TinyBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(312, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)

# class TextBiLSTM(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
#         super(TextBiLSTM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
        
#         self.lstm = nn.LSTM(embed_dim, 
#                             hidden_dim, 
#                             num_layers=n_layers, 
#                             bidirectional=True, 
#                             dropout=dropout if n_layers > 1 else 0,
#                             batch_first=True)
        
#         self.dropout = nn.Dropout(dropout)
#         # Hidden * 2 because it is Bidirectional
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
#     def forward(self, text):
#         # text = [batch size, sent len]
#         embedded = self.embedding(text)
        
#         # packed output, (hidden, cell)
#         output, (hidden, cell) = self.lstm(embedded)
        
#         # Concatenate the final forward and backward hidden states
#         hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
#         return self.fc(hidden)