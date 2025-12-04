from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Class untuk Hierarchical BERT dengan BiLSTM-Attention Gated Mechanism ============
class AttnGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

        init.xavier_normal_(self.Wr.state_dict()['weight'])
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        init.xavier_normal_(self.W.state_dict()['weight'])
        init.xavier_normal_(self.U.state_dict()['weight'])

    def forward(self, c, hi_1, g):
        r_i = torch.sigmoid(self.Wr(c) + self.Ur(hi_1))
        h_tilda = torch.tanh(self.W(c) + r_i * self.U(hi_1))
        hi = g * h_tilda + (1 - g) * hi_1
        return hi

class AttGRU(nn.Module):
    def __init__(self, bidirectional=True):
        super(AttGRU, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = AttnGRUCell(768, 768)
        if self.bidirectional:
            self.rnn_bwd = AttnGRUCell(768, 768)

    def forward(self, context, init_hidden, att_score, attn_mask=None):
        hidden = init_hidden
        if self.bidirectional:
            hidden, hidden_bwd = init_hidden.unsqueeze(1).transpose(0, 1).contiguous(), init_hidden.unsqueeze(1).transpose(0, 1).contiguous()
        inp = context.transpose(0, 1).contiguous()
        gates = att_score.unsqueeze(1)
        gates = gates.transpose(1, 2).transpose(0, 1).contiguous()

        seq_len = context.size()[1]

        for i in range(seq_len):
            hidden = self.rnn(inp[i:i + 1], hidden, gates[i:i + 1])

            if self.bidirectional:
                hidden_bwd = self.rnn_bwd(inp[seq_len - i - 1:seq_len - i], hidden_bwd,
                                          gates[seq_len - i - 1:seq_len - i])

        output = hidden.transpose(0, 1).contiguous()

        if self.bidirectional:
            output = torch.cat([hidden, hidden_bwd], dim=-1).transpose(0, 1).contiguous()  # batch x 1 x d_h*2
        return output

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.pooling_linear = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        weights = self.pooling_linear(x).squeeze(dim=2)
        if mask is not None:
            mask_value = torch.finfo(weights.dtype).min
            weights = weights.masked_fill(mask == 0, mask_value)
            # weights = weights.masked_fill(mask == 0, -1e9)
        att_score = nn.Softmax(dim=-1)(weights)

        return att_score

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0  # reset counter if improvement
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop_training = True
        return self.stop_training

class MeanPooling(nn.Module):
    def _init_(self):
        super(MeanPooling, self)._init_()

    def forward(self, embeddings, mask):
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        summed = torch.sum(embeddings * mask, dim=2)
        summed_mask = torch.clamp(mask.sum(dim=2), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled

class AGMencoder(nn.Module):
    def __init__(self, num_classes, hidden_size=768, hops=1, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.hops = hops

        # Define layers
        self.bert = BertModel.from_pretrained("indolem/indobert-base-uncased")
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=2,
                    dropout=0.2, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.att = SelfAttention(input_dim=hidden_size)
        self.mean_pooling = MeanPooling()

        self.AttGRUs = AttGRU()
        self.dropout_mid = nn.Dropout(dropout_rate)
        self.liner = nn.Linear(hidden_size * 2, 768)
        self.classifier = nn.Linear(768, num_classes)

    def init_hidden(self, batch_size, d_model, device=device):
        return Variable(torch.zeros(batch_size, d_model)).to(device)

    def forward(self, batch):
        device = next(self.parameters()).device  # Ambil device dari model
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)

        batch_size, num_containers, seq_length = input_ids.size()

        # Flatten input untuk masuk ke BERT
        input_ids_2d = input_ids.view(-1, seq_length).to(device)
        attention_masks_2d = attention_masks.view(-1, seq_length).to(device)
        # print(attention_masks_2d.shape)

        with torch.amp.autocast("cuda"):
            bert_output = self.bert(
                input_ids=input_ids_2d,
                attention_mask=attention_masks_2d,
            )
            embeddings_shape = bert_output.last_hidden_state

            embeddings = embeddings_shape.view(batch_size, num_containers, seq_length, self.hidden_size)

            attention_mask_reshaped = attention_masks.view(batch_size, num_containers, seq_length).to(device)

            # mean_pooled = self.mean_pooling(embeddings, attention_mask_reshaped)

            cls_tokens = embeddings[:, :, 0, :]
            cls_tokens = cls_tokens.view(batch_size, num_containers, self.hidden_size)


            if not hasattr(self, '_flattened'):
                self.lstm.flatten_parameters()
                setattr(self, '_flattened', True)
            hidden_state = self.dropout_mid(self.lstm(cls_tokens)[0])
            hidden_states = hidden_state + cls_tokens
            # print("hiden states setelah lstm", hidden_states.shape)

            container_mask = (attention_mask_reshaped.sum(dim=2) > 0).float()  # shape: (B, N)
            # print("container_mask", container_mask.shape)
            att_score = self.att(hidden_states,container_mask)

            s_out = []
            for hop in range(self.hops):
                attn_hid = self.init_hidden(hidden_states.size(0),hidden_states.size(-1))

                out_put = self.AttGRUs(hidden_states,attn_hid,att_score,container_mask)

                s_out.append(out_put)

            s_cont = torch.cat(s_out, dim=-1).squeeze(1)

            s_cont=self.dropout_mid(nn.ReLU()(self.liner(s_cont)))
            logits = self.classifier(s_cont)

        return logits