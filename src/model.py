from typing import List
from transformers import RobertaPreTrainedModel, RobertaConfig, RobertaModel
import torch
import torch.nn as nn


class LocateAndLabel(RobertaPreTrainedModel):
    """Model Locate and label"""

    def __init__(self, config: RobertaConfig, embed: torch.tensor, dropout: float, freeze_transformers: bool,
                 lstm_layers: int, lstm_dropout: float, pos_size: int, char_lstm_layers: int, char_lstm_dropout: float,
                 char_size: int, use_fasttext: bool, use_pos: bool, use_char_lstm: bool, n_poses: int,
                 list_char: List[str], n_entity_types: int, spn_filter: float):
        super(LocateAndLabel, self).__init__(config)

        self.bert = RobertaModel(config)
        self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_pos = use_pos
        self.use_fasttext = use_fasttext
        self.use_char_lstm = use_char_lstm
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_dropout = char_lstm_dropout
        self.char_size = char_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.freeze_transformers = freeze_transformers
        self.dropout = nn.Dropout(dropout)
        self.spn_filter = spn_filter

        lstm_input_size = config.hidden_size
        if self.use_fasttext:
            lstm_input_size += self.wordvec_size
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed)
        if self.use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(num_embeddings=n_poses, embedding_dim=self.pos_size)
        if self.use_char_lstm:
            self.list_char = list_char
            n_chars = len(self.list_char)
            lstm_input_size += self.char_size * 2
            self.char_embedding = nn.Embedding(num_embeddings=n_chars, embedding_dim=self.char_size)
            self.char_lstm = nn.LSTM(input_size=self.char_size, hidden_size=self.char_size,
                                     num_layers=self.char_lstm_layers, bidirectional=True,
                                     dropout=self.char_lstm_dropout, batch_first=True)

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=config.hidden_size // 2,
                            num_layers=self.lstm_layers, bidirectional=True,
                            dropout=self.lstm_dropout, batch_first=True)

        if self.freeze_transformers:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        self.binary_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
        )

        self.offset_regressor = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
            nn.Sigmoid()
        )

        self.entity_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, n_entity_types)
        )

        self.init_weights()

    def forward(self, encoding: torch.tensor, context_masks: torch.tensor, token_masks: torch.tensor,
                token_masks_bool: torch.tensor, entity_masks_token: torch.tensor,
                entity_spans_token: torch.tensor, pos_encoding: torch.tensor = None,
                wordvec_encoding: torch.tensor = None, char_encoding: torch.tensor = None,
                token_masks_char: torch.tensor = None, char_count: torch.tensor = None, mode: str = 'train'):
        r"""
        Forward function of Locate and Label model
        :param encoding:
        :param context_masks:
        :param token_masks:
        :param token_masks_bool:
        :param entity_masks_token:
        :param entity_spans_token:
        :param pos_encoding:
        :param wordvec_encoding:
        :param char_encoding:
        :param token_masks_char:
        :param char_count:
        :param mode:
        :return:
            entity_clf: outputs tensor of entity type classification ([batch_size, n_spans, n_entity_types])
            bin_clf: outputs tensor of filter span proposal ([batch_size, n_spans, 2])
            offsets: outputs tensor of boundary regressor ([batch_size, n_spans, 2])
            spn_mask: mark which seed spans are classified as positive
            illegal_mask: mark seed spans which after adjusting the boundary become meaningless (right > token_count,
            left < 0, right <= left)
        """
        context_masks = context_masks.float()
        batch_size = encoding.shape[0]
        token_count = token_masks_bool.long().sum(-1, keepdim=True)

        bert_outputs = self.bert(input_ids=encoding, attention_mask=context_masks)[0]
        h_token = combine(bert_outputs, token_masks, 'max')

        embeds = [h_token]
        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_fasttext:
            word_embed = self.wordvec_embedding(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)
        if self.use_char_lstm:
            char_count = char_count.view(-1)
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count * batch_size, max_char_count)
            char_encoding[char_count == 0][:, 0] = self.list_char.index('<EOT>')
            char_count[char_count == 0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input=char_embed, lengths=char_count.tolist(),
                                                                  enforce_sorted=False, batch_first=True)
            char_embed_packed_o, _ = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = combine(char_embed, token_masks_char, 'max')
            embeds.append(h_token_char)

        if len(embeds) > 1:
            h_token_pos_wordvec_char = torch.cat(embeds, dim=-1)
            h_token_pos_wordvec_char_packed = nn.utils.rnn.pack_padded_sequence(
                input=h_token_pos_wordvec_char,
                lengths=token_count.squeeze(-1).cpu().tolist(),
                enforce_sorted=False, batch_first=True
            )
            h_token_pos_wordvec_char_packed_o, _ = self.lstm(h_token_pos_wordvec_char_packed)
            h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token_pos_wordvec_char_packed_o, batch_first=True)

        bin_clf = self._filter_span_proposal(h_token, entity_masks_token, entity_spans_token)
        spn_p = torch.softmax(bin_clf, dim=-1)
        spn_mask = spn_p[:, :, 0] < self.spn_filter * spn_p[:, :, 1]

        offsets = self._adjust_boundary(h_token, entity_masks_token, entity_spans_token, token_count)
        offsets = offsets * token_count.unsqueeze(-2).expand(-1, offsets.size(1), 2)
        illegal_mask = torch.ones_like(spn_mask, device=spn_mask.device)
        old_entity_spans_token = entity_spans_token
        entity_spans_token = entity_spans_token + torch.round(offsets).to(dtype=torch.long)
        offsets[entity_spans_token[:, :, 0] >= entity_spans_token[:, :, 1]] = 0
        offsets[entity_spans_token[:, :, 0] < 0] = 0
        offsets[entity_spans_token[:, :, 1] > token_count] = 0

        illegal_mask[entity_spans_token[:, :, 0] >= entity_spans_token[:, :, 1]] = 0
        illegal_mask[entity_spans_token[:, :, 0] < 0] = 0
        illegal_mask[entity_spans_token[:, :, 1] > token_count] = 0
        entity_spans_token[entity_spans_token[:, :, 0] >= entity_spans_token[:, :, 1]] = old_entity_spans_token[
            entity_spans_token[:, :, 0] >= entity_spans_token[:, :, 1]]
        entity_spans_token[entity_spans_token[:, :, 0] < 0] = old_entity_spans_token[entity_spans_token[:, :, 0] < 0]
        entity_spans_token[entity_spans_token[:, :, 1] > token_count] = old_entity_spans_token[
            entity_spans_token[:, :, 1] > token_count]
        entity_masks_token = change_span_token_mask(entity_spans_token, entity_masks_token)

        entity_clf = self._classify_entity(h_token, entity_masks_token, entity_spans_token)

        return entity_clf, bin_clf, offsets, spn_mask, illegal_mask

    def _filter_span_proposal(self, h_token, entity_masks_token, entity_spans_token):
        batch_size = entity_spans_token.shape[0]
        entity_spans_pool = combine(h_token, entity_masks_token, 'max')
        entity_spans_token_inner = entity_spans_token.clone()
        entity_spans_token_inner[:, :, 0] = entity_spans_token_inner[:, :, 0]
        entity_spans_token_inner[:, :, 1] = entity_spans_token_inner[:, :, 1] - 1
        entity_spans_token_inner[:, :, 1][entity_spans_token[:, :, 1] < 0] = 0
        start_end_embedding_inner = torch.stack([h_token[i][entity_spans_token_inner[i]] for i in range(batch_size)])
        start_end_embedding_inner = start_end_embedding_inner.view(batch_size, start_end_embedding_inner.size(1), -1)
        embed_inner = [entity_spans_pool, start_end_embedding_inner]
        entity_repr_inner = torch.cat(embed_inner, dim=2)
        entity_repr_inner = self.dropout(entity_repr_inner)
        entity_clf = self.binary_classifier(entity_repr_inner)

        return entity_clf

    def _adjust_boundary(self, h_token, entity_masks_token, entity_spans_token, token_count):
        batch_size = entity_spans_token.shape[0]
        entity_spans_pool = combine(h_token, entity_masks_token, 'max')
        entity_spans_token_outer = entity_spans_token.clone()
        entity_spans_token_outer[:, :, 0] = entity_spans_token_outer[:, :, 0] - 1
        entity_spans_token_outer[:, :, 1] = entity_spans_token_outer[:, :, 1]
        entity_spans_token_outer[:, :, 0][entity_spans_token_outer[:, :, 0] < 0] = 0
        entity_spans_token_outer[:, :, 1][entity_spans_token_outer[:, :, 1] == token_count] = \
            token_count.repeat(1, entity_spans_token_outer.size(1))[
                entity_spans_token_outer[:, :, 1] == token_count] - 1
        start_end_embedding_outer = torch.stack([h_token[i][entity_spans_token_outer[i]] for i in range(batch_size)])
        start_end_embedding_outer = start_end_embedding_outer.view(batch_size, start_end_embedding_outer.size(1), -1)
        embed_outer = [entity_spans_pool, start_end_embedding_outer]
        entity_repr_outer = torch.cat(embed_outer, dim=2)
        entity_repr_outer = self.dropout(entity_repr_outer)
        offsets = self.offset_regressor(entity_repr_outer)

        return offsets

    def _classify_entity(self, h_token, entity_masks_token, entity_spans_token):
        batch_size = entity_spans_token.shape[0]
        entity_spans_pool = combine(h_token, entity_masks_token, 'max')
        entity_spans_token_inner = entity_spans_token.clone()
        entity_spans_token_inner[:, :, 0] = entity_spans_token_inner[:, :, 0]
        entity_spans_token_inner[:, :, 1] = entity_spans_token_inner[:, :, 1] - 1
        entity_spans_token_inner[:, :, 1][entity_spans_token_inner[:, :, 1] < 0] = 0
        start_end_embedding_inner = torch.stack([h_token[i][entity_spans_token_inner[i]] for i in range(batch_size)])
        start_end_embedding_inner = start_end_embedding_inner.view(batch_size, start_end_embedding_inner.size(1), -1)
        embed_inner = [entity_spans_pool, start_end_embedding_inner]
        entity_repr_inner = torch.cat(embed_inner, dim=2)
        entity_repr_inner = self.dropout(entity_repr_inner)
        entity_clf = self.entity_classifier(entity_repr_inner)

        return entity_clf


def combine(sub, sup_mask, pool_type='max'):
    """Combine different level representations"""

    sup = None
    if len(sub.shape) == len(sup_mask.shape):
        if pool_type == 'mean':
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2) / size
        elif pool_type == 'sum':
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2)
        elif pool_type == 'max':
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    else:
        if pool_type == 'mean':
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2) / size
        elif pool_type == 'sum':
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2)
        elif pool_type == 'max':
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0

    return sup


def change_span_token_mask(entity_spans_token, old_entity_masks_token):
    mask = torch.zeros(old_entity_masks_token.size(), dtype=torch.bool).to(device=old_entity_masks_token.device)
    for i, sample_entity_spans_token in enumerate(entity_spans_token):
        for j, span in enumerate(sample_entity_spans_token):
            mask[i, j, span[0]:span[1]] = 1

    return mask
