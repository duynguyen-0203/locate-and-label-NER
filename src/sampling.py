import torch
from typing import List
from src import utils
import random


def create_train_sample(doc, neg_entity_count: int, window_sizes: List[int], iou_spn: float):
    r"""
    Create a training sample for the model
    :param doc: document to build training sample
    :param neg_entity_count: the number of negative spans corresponding to one positive span. If a document doesn't have
    positive span, the number of negative samples is equal to 2
    :param window_sizes: the list of integers is the window sizes of the seed span
    :param iou_spn: the threshold for classifying a seed span as positive or negative
    :return: dictionary:
        encoding: token indices, numerical representations of tokens building the sequences that will be used as input
        by the BERT model
        pos_encoding: input of Pos embedding
        char_encoding: input of LSTM model for character embedding
        context_masks: attention mask for the BERT model
        token_mask_bool: mask the number of tokens in the document
        token_masks: mask the encoding position of each token in the document's encoding
        token_masks_char: mask the encoding of each character in the document's character level encoding
        char_count: the list of number of characters in each token (include <EOT> token)
        wordvec_encoding: the list of vocabulary id of tokens
        ious: list of iou scores of each seed span with its paired ground-truth entity, is the weight of each span when
        calculating loss (iou for positive span and 1 - iou for negative span)
        entity_types: list of entity types of each seed span (is entity types of ground-truth entity for positive span
        or NONE label for negative span)
        l_offsets: list of left offsets of each seed span (0 for negative span)
        r_offsets: list of right offsets of each seed span (0 for negative span)
        entity_sample_masks: mask the number of seed spans of the document (use when padding to the batch)
        offset_sample_masks: mask each seed span is positive or negative
        entity_masks_token: mask the list of tokens in each seed span
        entity_spans_token: span token of each seed span
        entity_masks: mask the list of encoding of all tokens in each seed span
    """
    encoding = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encoding)

    list_char_encoding = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in list_char_encoding:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token, dtype=torch.long))
    char_encoding = utils.padded_stack(char_encoding)
    token_masks_char = (char_encoding != 0).long()
    char_count = torch.tensor(char_count, dtype=torch.long)

    pos_encoding = [token.pos for token in doc.tokens]
    wordvec_encoding = [token.vocab_id for token in doc.tokens]

    token_masks = []
    for token in doc.tokens:
        token_masks.append(create_mask(*token.span, context_size))

    gold_entity_spans_token = []
    gold_entity_types = []
    gold_entity_spans = []
    for entity in doc.entities:
        gold_entity_spans_token.append(entity.span_token)
        gold_entity_types.append(entity.entity_type.id)
        gold_entity_spans.append(entity.span)

    pos_ious, pos_l_offsets, pos_r_offsets, pos_offset_sample_masks = [], [], [], []
    neg_ious, neg_l_offsets, neg_r_offsets, neg_offset_sample_masks = [], [], [], []

    pos_entity_spans, pos_entity_types, pos_entity_masks = [], [], []
    pos_entity_spans_token, pos_entity_masks_token = [], []
    neg_entity_spans, neg_entity_types, neg_entity_masks = [], [], []
    neg_entity_spans_token, neg_entity_masks_token = [], []

    for window_size in window_sizes:
        for i in range(0, token_count):
            left = i
            right = min(token_count, i + window_size + 1)
            span = doc.tokens[left:right].span
            span_token = doc.tokens[left:right].span_token
            if span_token not in pos_entity_spans_token and span_token not in neg_entity_spans_token:
                max_iou = 0
                entity_type, left_offset, right_offset = 0, 0, 0
                for j, gold_entity_span_token in enumerate(gold_entity_spans_token):
                    iou = utils.compute_iou(span_token, gold_entity_span_token)
                    if iou > max_iou:
                        max_iou = iou
                        entity_type = gold_entity_types[j]
                        left_offset = gold_entity_span_token[0] - span_token[0]
                        right_offset = gold_entity_span_token[1] - span_token[1]

                if max_iou > iou_spn:
                    pos_ious.append(max_iou)
                    pos_entity_types.append(entity_type)
                    pos_entity_spans.append(span)
                    pos_entity_spans_token.append(span_token)
                    pos_l_offsets.append(left_offset)
                    pos_r_offsets.append(right_offset)
                    pos_offset_sample_masks.append(1)
                else:
                    neg_ious.append(1 - max_iou)
                    neg_entity_types.append(0)
                    neg_entity_spans.append(span)
                    neg_entity_spans_token.append(span_token)
                    neg_l_offsets.append(0)
                    neg_r_offsets.append(0)
                    neg_offset_sample_masks.append(0)

    for i, gold_entity_span_token in enumerate(gold_entity_spans_token):
        if gold_entity_span_token not in pos_entity_spans_token:
            pos_ious.append(1)
            pos_entity_types.append(gold_entity_types[i])
            pos_entity_spans.append(gold_entity_spans[i])
            pos_entity_spans_token.append(gold_entity_span_token)
            pos_l_offsets.append(0)
            pos_r_offsets.append(0)
            pos_offset_sample_masks.append(1)

    # sample negative spans
    neg_entity_count = neg_entity_count * len(pos_entity_spans_token) + 2
    neg_entity_samples = random.sample(list(zip(neg_ious, neg_entity_types, neg_entity_spans, neg_entity_spans_token,
                                                neg_l_offsets, neg_r_offsets, neg_offset_sample_masks)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_ious, neg_entity_types, neg_entity_spans, neg_entity_spans_token, \
    neg_l_offsets, neg_r_offsets, neg_offset_sample_masks = map(list, zip(*neg_entity_samples) if neg_entity_samples
    else ([], [], [], [], [], [], []))

    pos_entity_masks_token = [create_mask(*span, token_count) for span in pos_entity_spans_token]
    pos_entity_masks = [create_mask(*span, context_size) for span in pos_entity_spans]
    neg_entity_masks_token = [create_mask(*span, token_count) for span in neg_entity_spans_token]
    neg_entity_masks = [create_mask(*span, context_size) for span in neg_entity_spans]

    # merge
    ious = pos_ious + neg_ious
    entity_types = pos_entity_types + neg_entity_types
    entity_types_1 = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    l_offsets = pos_l_offsets + neg_l_offsets
    r_offsets = pos_r_offsets + neg_r_offsets
    offset_sample_masks = pos_offset_sample_masks + neg_offset_sample_masks
    entity_masks_token = pos_entity_masks_token + neg_entity_masks_token
    entity_spans_token = pos_entity_spans_token + neg_entity_spans_token

    encoding = torch.tensor(encoding, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    token_masks = torch.stack(token_masks)

    ious = torch.tensor(ious, dtype=torch.float)
    entity_types = torch.tensor(entity_types, dtype=torch.long)
    entity_types_1 = torch.tensor(entity_types_1, dtype=torch.long)
    entity_masks = torch.stack(entity_masks)
    l_offsets = torch.tensor(l_offsets, dtype=torch.float)
    r_offsets = torch.tensor(r_offsets, dtype=torch.float)
    entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    offset_sample_masks = torch.tensor(offset_sample_masks, dtype=torch.bool)
    entity_masks_token = torch.stack(entity_masks_token)
    entity_spans_token = torch.tensor(entity_spans_token)

    return dict(encoding=encoding, pos_encoding=pos_encoding, char_encoding=char_encoding, context_masks=context_masks,
                token_masks_bool=token_masks_bool, token_masks=token_masks, token_masks_char=token_masks_char,
                char_count=char_count, wordvec_encoding=wordvec_encoding, ious=ious, entity_types=entity_types,
                l_offsets=l_offsets, r_offsets=r_offsets, entity_sample_masks=entity_sample_masks,
                offset_sample_masks=offset_sample_masks, entity_masks_token=entity_masks_token,
                entity_spans_token=entity_spans_token, entity_masks=entity_masks, entity_types_1=entity_types_1)


def create_eval_sample(doc, window_sizes: List[int]):
    r"""
    Create sample for evaluation phase
    :param doc: document to build training sample
    :param window_sizes: the list of integers is the window sizes of the seed span
    :return: dictionary:
        encoding: token indices, numerical representations of tokens building the sequences that will be used as input
        by the BERT model
        pos_encoding: input of Pos embedding
        char_encoding: input of LSTM model for character embedding
        context_masks: attention mask for the BERT model
        token_mask_bool: mask the number of tokens in the document
        token_masks: mask the encoding position of each token in the document's encoding
        token_masks_char: mask the encoding of each character in the document's character level encoding
        char_count: the list of number of characters in each token (include <EOT> token)
        wordvec_encoding: the list of vocabulary id of tokens
        entity_masks: mask the list of encoding of all tokens in each seed span
        entity_masks_token: mask the list of tokens in each seed span
        entity_spans: span in encoding of each seed span
        entity_spans_token: span token of each seed span
        entity_sample_masks: mask the number of seed spans of the document (use when padding to the batch)
    """
    encoding = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encoding)

    list_char_encoding = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in list_char_encoding:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token, dtype=torch.long))
    char_encoding = utils.padded_stack(char_encoding)
    token_masks_char = (char_encoding != 0).long()
    char_count = torch.tensor(char_count, dtype=torch.long)

    pos_encoding = [token.pos for token in doc.tokens]
    wordvec_encoding = [token.vocab_id for token in doc.tokens]

    token_masks = []
    for token in doc.tokens:
        token_masks.append(create_mask(*token.span, context_size))

    entity_spans, entity_masks, entity_spans_token, entity_masks_token = [], [], [], []

    for window_size in window_sizes:
        for i in range(0, token_count):
            left = i
            right = min(token_count, left + window_size + 1)
            span = doc.tokens[left:right].span
            span_token = doc.tokens[left:right].span_token
            if span not in entity_spans:
                entity_spans.append(span)
                entity_spans_token.append(span_token)
                entity_masks.append(create_mask(*span, context_size))
                entity_masks_token.append(create_mask(*span_token, token_count))

    encoding = torch.tensor(encoding, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    token_masks = torch.stack(token_masks)

    entity_masks = torch.stack(entity_masks)
    entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    entity_masks_token = torch.stack(entity_masks_token)
    entity_spans = torch.tensor(entity_spans, dtype=torch.long)
    entity_spans_token = torch.tensor(entity_spans_token, dtype=torch.long)

    return dict(encoding=encoding, pos_encoding=pos_encoding, wordvec_encoding=wordvec_encoding,
                char_encoding=char_encoding, char_count=char_count, token_masks_char=token_masks_char,
                context_masks=context_masks, token_masks_bool=token_masks_bool, token_masks=token_masks,
                entity_masks=entity_masks, entity_masks_token=entity_masks_token, entity_spans=entity_spans,
                entity_spans_token=entity_spans_token, entity_sample_masks=entity_sample_masks)


def create_mask(left, right, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[left:right] = 1

    return mask


def collate_fn(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = utils.padded_stack([s[key] for s in batch])

    return padded_batch
