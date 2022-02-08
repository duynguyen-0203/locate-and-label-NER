import json
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer
from src.entities import *

list_ner = ['LOC', 'MISC', 'ORG', 'PER']
list_pos = ['A', 'C', 'CH', 'Cc', 'E', 'FW', 'I', 'L', 'M', 'N', 'NNP', 'Nb', 'Nc', 'Ne', 'Ni', 'Ns', 'Nu', 'Ny', 'P',
            'R', 'T', 'V', 'Vb', 'X', 'Z']
list_char = ['!', '"', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
             '8', '9', ':', ';', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '°',
             '²', '³', '¼', 'À', 'Á', 'Â', 'Ê', 'Í', 'Ð', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê',
             'ì', 'í', 'ð', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'Ă', 'ă', 'Đ', 'đ', 'ĩ', 'Ũ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư',
             'Ạ', 'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'ẫ', 'Ậ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'Ặ', 'ặ', 'ẹ', 'ẻ',
             'ẽ', 'ế', 'ề', 'ể', 'Ễ', 'ễ', 'Ệ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ', 'ổ', 'Ỗ', 'ỗ', 'ộ',
             'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'ỡ', 'ợ', 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ',
             'ỵ', 'ỷ', 'ỹ', '–', '‘', '’', '“', '”', '…']


class Reader:
    def __init__(self, tokenizer: AutoTokenizer, vocab_path: str, embed_path: str, neg_entity_count: int,
                 window_sizes: List[int], iou_spn: float):
        self._tokenizer = tokenizer
        self._neg_entity_count = neg_entity_count
        self._window_sizes = window_sizes
        self._iou_spn = iou_spn

        self.list_pos = list_pos
        self.list_char = ['<PAD>'] + list_char + ['<EOT>', '<UNK>']

        self.entity_types = []
        none_entity_type = EntityType(0, 'NONE')
        self.entity_types.append(none_entity_type)
        for ner in list_ner:
            entity_type = EntityType(len(self.entity_types), ner)
            self.entity_types.append(entity_type)
        self.list_ner = list_ner

        self.word2inx = json.load(open(vocab_path, 'r', encoding='utf-8'))
        self.embedding_weight = np.load(embed_path)

    def read(self, dataset_path: str, name: str):
        dataset = Dataset(name, self.entity_types, self._neg_entity_count, self._window_sizes, self._iou_spn)
        with open(dataset_path, mode='r', encoding='utf-8') as f:
            documents = json.load(f)
        for document in tqdm(documents, total=len(documents), desc=f'Parse {name} dataset'):
            self._parse_document(document, dataset)

        return dataset

    def _parse_document(self, doc: dict, dataset):
        ltokens = doc['ltokens']
        rtokens = doc['rtokens']
        tokens = doc['tokens']
        entities = doc['entities']
        poses = doc['pos']
        name = doc['org_id']

        doc_encoding, doc_tokens, char_encoding = self._parse_tokens(ltokens, tokens, rtokens, poses, dataset)
        doc_entities = self._parse_entities(doc_tokens, entities, dataset)
        dataset.add_document(doc_tokens, doc_entities, doc_encoding, char_encoding, name)

    def _parse_tokens(self, jtokens: List[str], tokens: List[str], rtokens: List[str],
                      poses: List[str], dataset: Dataset):
        doc_encoding = [self._tokenizer.cls_token_id]
        char_encoding = []
        doc_tokens = []
        poses_id = [list_pos.index(pos) if pos in self.list_pos else self.list_pos.index('X') for pos in poses]

        for token in jtokens:
            doc_encoding.extend(self._tokenizer.encode(token, add_special_tokens=False))

        for i, token in enumerate(tokens):
            token_encoding = self._tokenizer.encode(token, add_special_tokens=False)
            span_start, span_end = len(doc_encoding), len(doc_encoding) + len(token_encoding)
            doc_encoding.extend(token_encoding)

            if token.lower() in self.word2inx.keys():
                inx = self.word2inx[token.lower()]
            else:
                inx = self.word2inx['<unk>']

            token_encoding_char = [self.list_char.index(c) if c in self.list_char
                                   else self.list_char.index('<UNK>') for c in token]
            token_encoding_char.append(self.list_char.index('<EOT>'))
            char_encoding.append(token_encoding_char)

            token_obj = dataset.add_token(i, token, span_start, span_end, poses_id[i], inx)
            doc_tokens.append(token_obj)

        for token in rtokens:
            doc_encoding.extend(self._tokenizer.encode(token, add_special_tokens=False))

        doc_encoding.append(self._tokenizer.sep_token_id)

        return doc_encoding, doc_tokens, char_encoding

    def _parse_entities(self, doc_token: List[Token], entities: List[dict], dataset: Dataset):
        doc_entities = []
        for entity in entities:
            start, end = entity['start'], entity['end']
            tokens = doc_token[start:end]
            entity_type = self.get_entity_type(entity['type'])
            entity = dataset.add_entity(entity_type, tokens)
            doc_entities.append(entity)

        return doc_entities

    def get_entity_type(self, index):
        if type(index) == str:
            for entity_type in self.entity_types:
                if entity_type.name == index:
                    return entity_type
        elif type(index) == int:
            for entity_type in self.entity_types:
                if entity_type.id == index:
                    return entity_type

        return self.get_entity_type('NONE')
