from typing import List
from collections import OrderedDict
from torch.utils.data import Dataset as TorchDataset
from src import sampling


class Token:
    def __init__(self, id: int, dataset_id: int, phrase: str, span_start: int, span_end: int, pos: int, vocab_id: int):
        self._id = id  # index of the token in the document
        self._dataset_id = dataset_id  # index of the token in the dataset
        self._phrase = phrase
        self._span_start = span_start  # start of token encoding in the document encoding (inclusive)
        self._span_end = span_end  # end of token encoding in the document encoding (exclusive)
        self._pos = pos
        self._vocab_id = vocab_id

    @property
    def id(self):
        return self._id

    @property
    def dataset_id(self):
        return self._dataset_id

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def pos(self):
        return self._pos

    @property
    def vocab_id(self):
        return self._vocab_id

    def __str__(self):
        return str(self._phrase)

    def __hash__(self):
        return hash(self.dataset_id)

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.dataset_id == other.dataset_id
        return False


class EntityType:
    def __init__(self, id: int, name: str):
        self._id = id  # dataset_id
        self._name = name

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


class TokenSpan:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens

    @property
    def tokens(self):
        return self._tokens

    @property
    def span_start(self):
        return self.tokens[0].span_start

    @property
    def span_end(self):
        return self.tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        return self.tokens[0].id, self.tokens[-1].id + 1

    def __str__(self):
        return ' '.join([str(t) for t in self.tokens])

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return TokenSpan(self.tokens[item.start:item.stop:item.step])
        else:
            return self.tokens[item]


class Entity:
    def __init__(self, id: int, entity_type: EntityType, tokens: List[Token]):
        self._id = id  # dataset_id
        self._entity_type = entity_type
        self._tokens = tokens

    def as_tuple_token(self):
        return self.span_token[0], self.span_token[1], self.entity_type.id

    @property
    def id(self):
        return self._id

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span(self):
        return self.tokens.span

    @property
    def span_token(self):
        return self.tokens.span_token

    def __str__(self):
        return ' '.join([str(t) for t in self.tokens])

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


class Document:
    def __init__(self, id: int, tokens: List[Token], entities: List[Entity], encoding: List[int],
                 char_encoding: List[List[int]], name: str = None):
        self._id = id
        self._tokens = tokens
        self._entities = entities
        self._encoding = encoding
        self._char_encoding = char_encoding
        self._name = name

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def encoding(self):
        return self._encoding

    @property
    def char_encoding(self):
        return self._char_encoding

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def entities(self):
        return self._entities

    def __len__(self):
        return len(self.tokens)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id
        return False

    def __str__(self):
        return str(self.tokens)

    def __hash__(self):
        return hash(self.id)


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, name: str, entity_types: List[EntityType], neg_entity_count: int, window_sizes: List[int],
                 iou_spn: float):
        self._mode = Dataset.TRAIN_MODE
        self._name = name
        self._entity_types = entity_types
        self._neg_entity_count = neg_entity_count
        self._window_sizes = window_sizes
        self._iou_spn = iou_spn

        self._documents = OrderedDict()
        self._entities = OrderedDict()

        self._token_id = 0
        self._doc_id = 0
        self._entity_id = 0

    def add_token(self, id: int, phrase: str, span_start: int, span_end: int, pos: int, vocab_id: int):
        token = Token(id, self._token_id, phrase, span_start, span_end, pos, vocab_id)
        self._token_id += 1

        return token

    def add_document(self, tokens: List[Token], entities: List[Entity], encoding: List[int],
                     char_encoding: List[List[int]], name: str = None):
        document = Document(self._doc_id, tokens, entities, encoding, char_encoding, name)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def add_entity(self, entity_type: EntityType, tokens: List[Token]):
        entity = Entity(self._entity_id, entity_type, tokens)
        self._entities[self._entity_id] = entity
        self._entity_id += 1

        return entity

    def set_mode(self, mode: str):
        self._mode = mode

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def entity_count(self):
        return len(self.entities)

    @property
    def documents(self):
        return list(self._documents.values())

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        if self._mode == 'train':
            return sampling.create_train_sample(self.documents[item], self._neg_entity_count, self._window_sizes,
                                                self._iou_spn)
        else:
            return sampling.create_eval_sample(self.documents[item], self._window_sizes)
