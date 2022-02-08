import torch
import transformers
from transformers import AutoTokenizer, RobertaConfig, AdamW
from datetime import datetime
import os
import logging
import sys
# from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import csv

from src import utils
from src import logger_utils
from src.input_reader import Reader
from src.entities import Dataset
from src.model import LocateAndLabel
from src.losses.focal_loss import FocalLoss
from src.losses.giou_loss import GiouLoss
from src.losses.loss import ModelLoss
from src.evaluation import Evaluator


class Trainer:
    def __init__(self, args):
        self.args = args
        self._init_logger()
        self._device = torch.device(utils.get_device())
        utils.set_seed(args.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=False)

    def train(self):
        args = self.args
        self._logger.info(f'Model: {args.model_name}')
        self._logger.info(f'Dataset: {args.data_name}')

        reader = Reader(self._tokenizer, args.vocab_path, args.embed_path, args.neg_entity_count, args.window_sizes,
                        args.iou_spn)
        train_dataset = reader.read(args.train_data_path, args.data_name)
        valid_dataset = reader.read(args.valid_data_path, args.data_name)
        self._log_dataset(reader, train_dataset, valid_dataset)
        n_train_samples = len(train_dataset)
        updates_epoch = n_train_samples // args.batch_size
        n_updates = updates_epoch * args.epochs
        self._logger.info(f'Updates per epoch: {updates_epoch}')
        self._logger.info(f'Updates total: {n_updates}')

        # create model
        config = RobertaConfig.from_pretrained(args.bert_path)
        embed = torch.from_numpy(reader.embedding_weight).float()
        model = LocateAndLabel.from_pretrained(args.bert_path, config=config, embed=embed, dropout=args.dropout,
                                               freeze_transformers=args.freeze_transformers,
                                               lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout,
                                               pos_size=args.pos_size, char_lstm_layers=args.char_lstm_layers,
                                               char_lstm_dropout=args.char_lstm_dropout, char_size=args.char_size,
                                               use_fasttext=args.use_fasttext, use_pos=args.use_pos,
                                               use_char_lstm=args.use_char_lstm, n_poses=len(reader.list_pos),
                                               list_char=reader.list_char, n_entity_types=len(reader.entity_types),
                                               spn_filter=args.spn_filter)
        model.to(self._device)
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.lr_warmup * n_updates,
                                                                 num_training_steps=n_updates)

        # create loss
        filter_criterion = FocalLoss(n_classes=2, reduction='none', gamma=args.filter_gamma)
        offset_criterion = torch.nn.SmoothL1Loss(reduction='none')
        giou_criterion = GiouLoss(task='giou', reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss_calculator = ModelLoss(filter_criterion, offset_criterion, giou_criterion, entity_criterion, model,
                                    optimizer, scheduler, args.max_grad_norm, args.filter_weight, args.offset_weight,
                                    args.giou_weight, args.entity_weight, args.iou_classifier, args.iou_eta)

        best_f1 = 0
        for epoch in range(args.epochs):
            epoch_loss = self._train_epoch(model, loss_calculator, optimizer, train_dataset, epoch)
            self._logger.info(f'Epoch loss {epoch_loss}')
            eval_result = self._eval_epoch(model, valid_dataset, epoch, reader)
            if eval_result['macro avg']['f1-score'] > best_f1:
                best_f1 = eval_result['macro avg']['f1-score']
                self._logger.info(f'New best model at epoch {epoch}.')
                self._save_model(model, optimizer, scheduler, epoch, flag='bestModel')
            self._log_csv([epoch_loss, eval_result['macro avg']['precision'], eval_result['macro avg']['recall'],
                           eval_result['macro avg']['f1-score']])

        self._save_model(model, optimizer, scheduler, args.epochs - 1, flag='finalModel')

    def _train_epoch(self, model: torch.nn.Module, loss_calculator: ModelLoss, optimizer: Optimizer, dataset: Dataset,
                     epoch: int):
        self._logger.info(f'-------------------EPOCH {epoch}-------------------')
        dataset.set_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                                 collate_fn=self._collate_fn)
        model.zero_grad()
        epoch_loss = 0.0
        for batch in tqdm(data_loader, total=len(data_loader), desc=f'Train epoch {epoch}'):
            model.train()
            batch = utils.to_device(batch, self._device)
            entity_clf, bin_clf, offsets, spn_mask, illegal_mask = model(encoding=batch['encoding'],
                                                                         context_masks=batch['context_masks'],
                                                                         token_masks=batch['token_masks'],
                                                                         token_masks_bool=batch['token_masks_bool'],
                                                                         entity_masks_token=batch['entity_masks_token'],
                                                                         entity_spans_token=batch['entity_spans_token'],
                                                                         pos_encoding=batch['pos_encoding'],
                                                                         wordvec_encoding=batch['wordvec_encoding'],
                                                                         char_encoding=batch['char_encoding'],
                                                                         token_masks_char=batch['token_masks_char'],
                                                                         char_count=batch['char_count'],
                                                                         mode='train')
            
            batch_loss = loss_calculator.compute(bin_logits=bin_clf, offsets=offsets, entity_logits=entity_clf,
                                                 illegal_mask=illegal_mask, gold_entity_types=batch['entity_types'],
                                                 gold_l_offsets=batch['l_offsets'], gold_r_offsets=batch['r_offsets'],
                                                 ious=batch['ious'], entity_sample_masks=batch['entity_sample_masks'],
                                                 offset_sample_masks=batch['offset_sample_masks'],
                                                 entity_spans_token=batch['entity_spans_token'])
            epoch_loss += (batch_loss / self.args.batch_size)

        return epoch_loss / len(data_loader)

    def _eval_epoch(self, model: torch.nn.Module, dataset: Dataset, epoch: int, reader: Reader):
        evaluator = Evaluator(dataset, reader, self.args.nms_decay, self.args.nms_shohold, self.args.nms)
        dataset.set_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self._collate_fn)
        with torch.no_grad():
            model.eval()
            total = math.ceil(len(dataset) / self.args.batch_size)
            for batch in tqdm(data_loader, total=total, desc=f'Evaluate epoch {epoch}'):
                batch = utils.to_device(batch, self._device)
                entity_clf, bin_clf, offsets, spn_mask, illegal_mask = model(encoding=batch['encoding'],
                                                                             context_masks=batch['context_masks'],
                                                                             token_masks=batch['token_masks'],
                                                                             token_masks_bool=batch['token_masks_bool'],
                                                                             entity_masks_token=batch[
                                                                                 'entity_masks_token'],
                                                                             entity_spans_token=batch[
                                                                                 'entity_spans_token'],
                                                                             pos_encoding=batch['pos_encoding'],
                                                                             wordvec_encoding=batch['wordvec_encoding'],
                                                                             char_encoding=batch['char_encoding'],
                                                                             token_masks_char=batch['token_masks_char'],
                                                                             char_count=batch['char_count'],
                                                                             mode='eval')
                evaluator.eval_batch(entity_clf, spn_mask, illegal_mask, offsets, batch)
        eval_result = evaluator.eval(print_result=True, list_ner=reader.list_ner)

        return eval_result

    def _init_logger(self):
        time = str(datetime.now()).replace(' ', '_').replace(':', '-')
        self._path = os.path.join(self.args.save_path, time)
        self._log_path = os.path.join(self._path, 'log')
        os.makedirs(self._path, exist_ok=True)
        os.makedirs(self._log_path, exist_ok=True)

        log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s [%(levelname)-5.5s] %(message)s')
        self._logger = logging.getLogger()
        logger_utils.reset_logger(self._logger)

        file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))
        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)

        self._logger.setLevel(logging.INFO)
        # self._summary_writer = SummaryWriter(self._log_path)
        self._init_csv_logger()
        self._csv_file = os.path.join(self._log_path, 'result.csv')

        self._log_arguments()

    def _log_arguments(self):
        logger_utils.save_dict(self._log_path, 'args', self.args)

    def _log_dataset(self, reader: Reader, train_dataset: Dataset, valid_dataset: Dataset):
        self._logger.info(f'NER label: {reader.list_ner}')
        self._logger.info(f'Train dataset: {len(train_dataset)} samples, {train_dataset.entity_count} entities')
        self._logger.info(f'Validation dataset: {len(valid_dataset)} samples, {valid_dataset.entity_count} entities')

    def _save_model(self, model: torch.nn.Module, optimizer: Optimizer, scheduler, epoch: int, flag: str):
        save_path = os.path.join(self._path, flag + '.pt')
        saved_point = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(saved_point, save_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        return optimizer_params

    def _collate_fn(self, batch):
        padded_batch = dict()
        keys = batch[0].keys()

        for key in keys:
            samples = [s[key] for s in batch]
            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                if key == 'encoding':
                    padded_batch[key] = utils.padded_stack([s[key] for s in batch],
                                                           padding=self._tokenizer.pad_token_id)
                else:
                    padded_batch[key] = utils.padded_stack([s[key] for s in batch])

        return padded_batch

    def _init_csv_logger(self):
        self._csv_file = os.path.join(self._log_path, 'result.csv')
        header = ['epoch', 'train_loss', 'val_f1_precision_score', 'val_f1_recall_score', 'val_f1_score']
        with open(self._csv_file, mode='w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def _log_csv(self, data):
        with open(self._csv_file, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
