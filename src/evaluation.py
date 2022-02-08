import torch
from typing import List
from sklearn.metrics import classification_report

from src.entities import Dataset, Document
from src.input_reader import Reader
from src import utils


class Evaluator:
    def __init__(self, dataset: Dataset, reader: Reader, nms_decay: float, nms_shohold: float, nms: float):
        self._dataset = dataset
        self._reader = reader
        self.pred_entities = []
        self.gold_entities = []
        self._convert_gold_entities(dataset.documents)
        self._nms_decay = nms_decay
        self._nms_shohold = nms_shohold
        self._nms = nms

    def eval_batch(self, batch_entity_clf: torch.tensor, spn_mask: torch.tensor, illegal_mask: torch.tensor,
                   offsets: torch.tensor, batch: dict):
        r"""
        :param batch_entity_clf: [batch_size, n_spans, n_entity_types]
        :param spn_mask: [batch_size, n_spans]
        :param illegal_mask: [batch_size, n_spans]
        :param offsets: [batch_size, n_spans, 2]
        :param batch:
        :return:
        """
        batch_size = batch_entity_clf.shape[0]
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        batch_entity_types *= (batch['entity_sample_masks'].long() * spn_mask.long() * illegal_mask.long())
        for i in range(batch_size):
            entity_types = batch_entity_types[i]
            valid_entity_indices = entity_types.nonzero(as_tuple=False).view(-1)
            valid_entity_types = entity_types[valid_entity_indices]

            old_entity_spans_token = batch['entity_spans_token'][i][valid_entity_indices]
            valid_entity_spans_token = old_entity_spans_token + offsets[i][valid_entity_indices].long()

            token_count = batch['token_masks_bool'][i].long().sum()
            valid_entity_spans_token[:, 0][valid_entity_spans_token[:, 0] < 0] = 0
            valid_entity_spans_token[:, 1][valid_entity_spans_token[:, 1] > token_count] = token_count
            valid_entity_spans_token[valid_entity_spans_token[:, 0] >= valid_entity_spans_token[:, 1]] \
                = old_entity_spans_token[valid_entity_spans_token[:, 0] >= valid_entity_spans_token[:, 1]]

            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)
            sample_preds = self._convert_pred_entities(valid_entity_types, valid_entity_spans_token,
                                                       valid_entity_scores)
            sample_preds = self._soft_nms(sample_preds)
            self.pred_entities.append(sample_preds)

    def eval(self, print_result, list_ner: List[str]):
        assert len(self.pred_entities) == len(self.gold_entities)
        gold_flat = []
        pred_flat = []

        for (sample_pred, sample_gold) in zip(self.pred_entities, self.gold_entities):
            union = set()
            union.update(sample_pred)
            union.update(sample_gold)

            for span in union:
                if span in sample_gold:
                    gold_flat.append(span[2])
                else:
                    gold_flat.append(0)

                if span in sample_pred:
                    pred_flat.append(span[2])
                else:
                    pred_flat.append(0)

        print(pred_flat)
        print(gold_flat)
        if print_result:
            print(classification_report(gold_flat, pred_flat, labels=list(range(1, len(list_ner) + 1)),
                                        target_names=list_ner, zero_division=0))

        return classification_report(gold_flat, pred_flat, labels=list(range(1, len(list_ner) + 1)),
                                     target_names=list_ner, digits=4, output_dict=True, zero_division=0)

    def _convert_gold_entities(self, docs: List[Document]):
        for doc in docs:
            doc_entities = [entity.as_tuple_token() for entity in doc.entities]
            self.gold_entities.append(doc_entities)

    @staticmethod
    def _convert_pred_entities(entity_types: torch.tensor, entity_spans_token: torch.tensor,
                               entity_scores: torch.tensor):
        r"""
        Convert the predictions of a sample
        :param entity_types: [n_valid_spans]
        :param entity_spans_token: [n_valid_spans, 2]
        :param entity_scores: [n_valid_spans, 2]
        :return: list of prediction in format (left, right, type, score)
        """
        preds = []
        unique = set()
        for i, (entity_type, entity_span_token, entity_score) in \
                enumerate(zip(entity_types, entity_spans_token, entity_scores)):
            if entity_span_token in unique:
                continue
            score = entity_score
            for j, (other_type, other_span_token, other_score) in \
                    enumerate(list(zip(entity_types, entity_spans_token, entity_scores))[i + 1:]):
                if entity_span_token[0] == other_span_token[0] and entity_span_token[1] == other_span_token[1] \
                        and entity_type == other_type:
                    score += other_score
            preds.append((entity_span_token[0].item(), entity_span_token[1].item(), entity_type.item(),
                          entity_score.item()))
            unique.add(entity_span_token)

        return preds

    def _soft_nms(self, preds):
        preds = sorted(preds, key=lambda x: x[3], reverse=True)
        preds = self._remove_partial_overlapping(preds)
        size = len(preds)
        for i, pred in enumerate(preds):
            start, end, _, score = pred
            for j in range(i + 1, size):
                other_start, other_end, other_type, other_score = preds[j]
                if utils.compute_iou((start, end), (other_start, other_end)) > self._nms_shohold:
                    preds[j] = (other_start, other_end, other_type, other_score * self._nms_decay)
                    not_insert = 1
                    for k in range(j + 1, size):
                        if other_score > preds[k][3]:
                            not_insert = 0
                            preds.insert(k, preds[j])
                            break
                    if not_insert:
                        preds.append(preds[j])
                    del preds[j]

        return list(filter(lambda x: x[3] > self._nms, preds))

    def _remove_partial_overlapping(self, preds):
        non_overlapping_entities = []
        for i, pred in enumerate(preds):
            if not self._is_partial_overlapping(pred, preds):
                non_overlapping_entities.append(pred)

        return non_overlapping_entities

    def _is_partial_overlapping(self, pred, preds):
        for other_pred in preds:
            if self._check_partial_overlap(pred, other_pred):
                return True

        return False

    @staticmethod
    def _check_partial_overlap(pred_1, pred_2):
        if (pred_1[0] < pred_2[0] < pred_1[1] < pred_2[1]) or (pred_2[0] < pred_1[0] < pred_2[1] < pred_1[1]):
            return True
        else:
            return False
