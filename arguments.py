import argparse
from typing import List


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Locate-and-label arguments", allow_abbrev=False)

    parser = _add_data_args(parser)
    parser = _add_model_args(parser)
    parser = _add_training_args(parser)
    parser = _add_loss_args(parser)
    parser = _add_evaluation_args(parser)

    return parser


def _add_data_args(parser):
    parser.add_argument('--data_name', type=str, default='VLSP_2016', help='Name of the dataset')
    parser.add_argument('--vocab_path', type=str, default='data/VLSP_2016/vocab.json', help='Path to the vocab file.')
    parser.add_argument('--embed_path', type=str, default='data/VLSP_2016/fasttext.npy', help='Path to the embed file.')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length to process')
    parser.add_argument('--train_data_path', type=str, default='data/VLSP_2016/train_data.json',
                        help='Path to the training dataset')
    parser.add_argument('--valid_data_path', type=str, default='data/VLSP_2016/dev_data.json',
                        help='Path to the validation dataset')
    parser.add_argument('--iou_spn', type=float, default=0.7)
    parser.add_argument('--neg_entity_count', type=int, default=5,
                        help='The number of negative spans corresponding to one positive span')
    parser.add_argument('--window_sizes', type=List[int], default=list(range(11)),
                        help='List of window sizes of seed span')

    return parser


def _add_model_args(parser):
    parser.add_argument('--model_name', type=str, default='Locate and Label', help='Name of the model')
    parser.add_argument('--spn_filter', type=float, default=5, help='Span proposal filter parameter')
    parser.add_argument('--bert_path', type=str, default='vinai/phobert-base', help='BERT Model')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--freeze_transformers', action='store_true', default=False)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--lstm_dropout', type=float, default=0.2)
    parser.add_argument('--pos_size', type=int, default=25)
    parser.add_argument('--char_lstm_layers', type=int, default=1)
    parser.add_argument('--char_lstm_dropout', type=float, default=0.2)
    parser.add_argument('--char_size', type=float, default=50)
    parser.add_argument('--use_fasttext', action='store_true', default=True)
    parser.add_argument('--use_pos', action='store_true', default=True)
    parser.add_argument('--use_char_lstm', action='store_true', default=True)
    parser.add_argument('--iou_classifier', type=float, default=1.0)
    parser.add_argument('--iou_eta', type=float, default=1.0)
    return parser


def _add_training_args(parser):
    parser.add_argument('--seed', type=int, default=100, help='Seed value')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_warmup', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, default='training')

    return parser


def _add_loss_args(parser):
    parser.add_argument('--filter_gamma', type=int, default=2)
    parser.add_argument('--filter_weight', type=float, default=1.0)
    parser.add_argument('--offset_weight', type=float, default=1.0)
    parser.add_argument('--giou_weight', type=float, default=1.0)
    parser.add_argument('--entity_weight', type=float, default=1.0)

    return parser


def _add_evaluation_args(parser):
    parser.add_argument('--nms', type=float, default=0.45,
                        help='Confidence score threshold to decide whether to keep a span proposal as an entity')
    parser.add_argument('--nms_shohold', type=float, default=0.6,
                        help='The IoU threshold to decide whether adjust the score of a span proposal')
    parser.add_argument('--nms_decay', type=float, default=0.9,
                        help='The decay coefficient of the score in Soft-NMS algorithm')

    return parser
