import os.path
import pickle
import string

import gensim
import h5py
import progressbar
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchtext

from .dataset import BachelorThesisDataset


class BachelorThesisData(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.data = h5py.File(kwargs['data_path'], 'r')
        self.word2vec_path = kwargs['word2vec_path']
        self.train_years = kwargs['train_years']
        self.validation_years = kwargs['validation_years']
        self.test_years = kwargs['test_years']
        self.dict_size = kwargs['dict_size']
        self.n_news = kwargs['n_news']
        self.n_words = kwargs['n_words']
        self.symbol = kwargs['symbol']
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.words_list, self.words_embeddings = None, None

    def prepare_data(self, data_ckpt_path='./lightning_logs/data.ckpt'):
        """Creates the dictionary along with its word embeddings.

        Stores as a class attribute a ``torch.Tensor`` of size (dict_size,
        embedding_size) with the Word2Vec embeddings used in a
        ``torch.nn.Embedding`` layer. Only the first ``dict_size`` words
        (ordered by frequency) are stored.

        Args:
            data_ckpt_path (str): path to the generated data file.
        """
        if os.path.exists(data_ckpt_path):
            with open(data_ckpt_path, 'rb') as data_ckpt:
                self.words_list, self.words_embeddings = pickle.load(data_ckpt)
        else:
            words_dict = self._get_words_frequency()
            self.words_list, self.words_embeddings = \
                self._get_words_list_embeddings(words_dict)
            with open(data_ckpt_path, 'wb') as data_ckpt:
                pickle.dump(
                    (self.words_list, self.words_embeddings), data_ckpt
                )

    def _get_words_frequency(self):
        words_dict = {}
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        bar = progressbar.ProgressBar(len(self.data['news/reuters'].keys()))
        for i, day in enumerate(self.data['news/reuters'].keys()):
            day_news = self.data['news/reuters'][day][()]
            for new_index in range(day_news.shape[0]):
                for new_word in tokenizer(
                        day_news[new_index, 0].decode('utf-8').translate(
                            str.maketrans('', '', string.punctuation)
                        )
                ):
                    words_dict[new_word] = words_dict[new_word] + 1 \
                        if new_word in words_dict else 1
            bar.update(i)
        return words_dict

    def _get_words_list_embeddings(self, words_dict):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            self.word2vec_path, binary=True
        )

        words_list = ['<EMP>', '<UNK>']
        words_embeddings = [torch.zeros((300,)), torch.ones((300,))]
        for word in sorted(
                words_dict.keys(), key=lambda x: words_dict[x], reverse=True
        ):
            if word in word2vec:
                words_list.append(word)
                words_embeddings.append(
                    torch.from_numpy(word2vec[word])
                )
            if len(words_list) >= self.dict_size:
                break
        return words_list, torch.stack(words_embeddings)

    def train_dataloader(self):
        train_dataset = BachelorThesisDataset(
            self.data, self.symbol, self.train_years, self.words_list,
            self.n_news, self.n_words
        )
        return torch.utils.data.DataLoader(
            train_dataset, self.batch_size, True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        validation_dataset = BachelorThesisDataset(
            self.data, self.symbol, self.validation_years, self.words_list,
            self.n_news, self.n_words
        )
        return torch.utils.data.DataLoader(
            validation_dataset, self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        test_dataset = BachelorThesisDataset(
            self.data, self.symbol, self.test_years, self.words_list,
            self.n_news, self.n_words
        )
        return torch.utils.data.DataLoader(
            test_dataset, self.batch_size, num_workers=self.num_workers
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('BachelorThesisData')
        parser.add_argument(
            '--data_path', default='./data/bachelor_thesis_data.hdf5'
        )
        parser.add_argument('--word2vec_path', default='./data/word2vec.bin')
        parser.add_argument('--dict_size', type=int, default=10000)
        parser.add_argument('--n_news', type=int, default=25)
        parser.add_argument('--n_words', type=int, default=15)
        parser.add_argument('--symbol', default='AAPL')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--train_years', type=int, nargs='+',
                            default=[2010, 2011, 2012, 2013, 2014, 2015, 2016])
        parser.add_argument('--validation_years', type=int, nargs='+',
                            default=[2017])
        parser.add_argument('--test_years', type=int, nargs='+',
                            default=[2018])
        return parent_parser
