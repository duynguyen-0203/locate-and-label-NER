from abc import ABC
import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class ModelLoss(Loss):
    def __init__(self, filter_criterion, offset_criterion, giou_criterion, entity_criterion, model, optimizer,
                 scheduler, max_grad_norm, filter_weight, offset_weight, giou_weight, entity_weight, iou_classifier,
                 iou_eta):
        r"""
        Initialization
        :param filter_criterion: the criterion to calculate the loss of span proposal filter
        :param offset_criterion: the criterion to calculate the loss at boundary level of boundary regressor
        :param giou_criterion: the criterion to calculate the overlap loss of boundary regressor
        :param entity_criterion: the criterion to calculate the loss of entity classifier module
        :param model:
        :param optimizer:
        :param scheduler:
        :param max_grad_norm: maximum gradient norm
        :param filter_weight: weight of filter loss
        :param offset_weight: weight of offset loss
        :param giou_weight: weight of giou loss
        :param entity_weight: weight of entity classifier loss
        :param iou_classifier: the threshold to reassign the categories based on the IoU between the new adjusted span
        proposal and paired ground-truth entity
        :param iou_eta: focusing parameter that adjust the weight of each span
        """
        self._filter_criterion = filter_criterion
        self._offset_criterion = offset_criterion
        self._giou_criterion = giou_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._filter_weight = filter_weight
        self._offset_weight = offset_weight
        self._giou_weight = giou_weight
        self._entity_weight = entity_weight
        self._iou_classifier = iou_classifier
        self._iou_eta = iou_eta

    def compute(self, bin_logits, offsets, entity_logits, illegal_mask, gold_entity_types, gold_l_offsets,
                gold_r_offsets, ious, entity_sample_masks, offset_sample_masks, entity_spans_token):
        r"""
        Compute training loss for Locate and Label model
        :param bin_logits: outputs tensor of filter span proposal
        :param offsets: outputs tensor of boundary regressor
        :param entity_logits: outputs tensor of entity type classification
        :param illegal_mask: mark seed spans which after adjusting the boundary become meaningless
        :param gold_entity_types: list of entity types of each seed span to predict
        :param gold_l_offsets: list of left offsets of each seed span to predict
        :param gold_r_offsets: list of right offsets of each seed span to predict
        :param ious: list of iou scores of each seed span with its paired ground-truth entity, is the weight of each
        span when calculating loss
        :param entity_sample_masks: mask the number of seed spans of the document (use when padding to the batch)
        :param offset_sample_masks: mask each seed span is positive or negative
        :param entity_spans_token: span token of each seed span
        :return:
        """
        old_ious = ious
        ious = ious.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()
        old_illegal_mask = illegal_mask
        illegal_mask = illegal_mask.view(-1).float()
        old_entity_types = gold_entity_types
        gold_entity_types = gold_entity_types.view(-1)
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])

        # compute loss of span proposal filter
        bin_logits = bin_logits.view(-1, bin_logits.shape[-1])
        bin_entity_types = (old_entity_types != 0).view(-1).to(dtype=torch.long)
        filter_loss = self._filter_criterion(bin_logits, bin_entity_types) * ious
        filter_loss = (filter_loss * entity_sample_masks).sum() / (entity_sample_masks.sum())

        # compute loss of boundary regressor
        giou_loss = self._giou_criterion((offsets + entity_spans_token).view(-1, 2),
                                         (torch.stack([gold_l_offsets, gold_r_offsets], dim=-1)
                                          + entity_spans_token).view(-1, 2))
        l_offsets_loss = self._offset_criterion(offsets[:, :, 0].squeeze(-1), gold_l_offsets) * old_ious
        r_offsets_loss = self._offset_criterion(offsets[:, :, 1].squeeze(-1), gold_r_offsets) * old_ious
        l_offsets_loss[torch.isinf(l_offsets_loss)] = 0
        r_offsets_loss[torch.isinf(r_offsets_loss)] = 0

        offset_sample_masks = offset_sample_masks * old_illegal_mask
        giou_loss = (giou_loss * offset_sample_masks.view(-1)).sum() / (offset_sample_masks.sum() + 1e-30)
        l_offsets_loss = (l_offsets_loss * offset_sample_masks).sum() / (offset_sample_masks.sum() + 1e-30)
        r_offsets_loss = (r_offsets_loss * offset_sample_masks).sum() / (offset_sample_masks.sum() + 1e-30)
        offsets_loss = l_offsets_loss + r_offsets_loss

        # compute loss of entity classifier module
        entity_sample_masks = entity_sample_masks * illegal_mask
        new_iou = compute_iou((torch.round(offsets) + entity_spans_token).view(-1, 2),
                              (torch.stack([gold_l_offsets, gold_r_offsets], dim=-1) + entity_spans_token).view(-1, 2))
        entity_types = gold_entity_types * (new_iou >= self._iou_classifier) * old_illegal_mask.long().view(-1)
        new_iou[new_iou < self._iou_classifier] = torch.pow(1 - new_iou[new_iou < self._iou_classifier],
                                                            self._iou_eta)
        new_iou[new_iou >= self._iou_classifier] = torch.pow(new_iou[new_iou >= self._iou_classifier], self._iou_eta)
        entity_loss = self._entity_criterion(entity_logits, entity_types) * new_iou
        entity_loss = (entity_loss * entity_sample_masks).sum() / (entity_sample_masks.sum() + 1e-30)

        train_loss = self._filter_weight * filter_loss + self._giou_weight * giou_loss + \
                     self._offset_weight * offsets_loss + self._entity_weight * entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()

        return train_loss.item()


def compute_iou(inputs, targets):
    max_left = torch.max(inputs[:, 0], targets[:, 0])
    min_right = torch.min(inputs[:, 1], targets[:, 1])
    max_right = torch.max(inputs[:, 1], targets[:, 1])
    min_left = torch.min(inputs[:, 0], targets[:, 0])

    all = max_right - min_left
    all[all == 0] = 1e-30
    iou = (min_right - max_left) / all
    iou[iou < 0] = 0

    return iou
