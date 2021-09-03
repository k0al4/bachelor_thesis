import random
import re
import string

import torch.nn.utils.rnn
import torch.utils.data
import torchtext


class BachelorThesisDataset(torch.utils.data.Dataset):

    def __init__(self, data, symbol, years, words_list, n_news, n_words):
        self.words_list = words_list
        self.n_news = n_news
        self.n_words = n_words
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        self._init_items_indexes(data, symbol, years)

    def _init_items_indexes(self, data, symbol, years):
        self.items_indexes = []
        news_items_list = list(data['news/reuters'].keys())
        items_cands = filter(
            lambda x: int(x[:4]) in years,
            data['prices/{}'.format(symbol)].keys()
        )
        for item in items_cands:
            if item in news_items_list and len(
                    data['news/reuters'][item][()]
            ) >= self.n_news:
                self.items_indexes.append(item)

        self.news, self.prices = [], []
        for item_index in self.items_indexes:
            self.news.append(data['news/reuters'][item_index][()])
            self.prices.append(
                data['prices/{}'.format(symbol)][item_index][()]
            )

    def __getitem__(self, item):
        """Returns the news and the movement to predict of a certain date.
        Returns a set of ``self.n_news`` news along with a binary value
        representing if the stock has increased or decreased its price during
        that date.

        Args:
            item (int): index of the date in ``self.items_indexes``.

        Returns:
            torch.Tensor: tensor of size (self.n_news, self.n_words) containing
                ``self.n_news`` random news of the date.
            int: 1 if the stock price has increased its price, 0 otherwise.
        """
        item_news, item_price = self.news[item], self.prices[item]
        news_titles = [
            item_news[new_index][0] for new_index in random.sample(
                list(range(item_news.shape[0])), self.n_news
            )
        ]
        return self._pack_news_titles(news_titles), \
            1 if item_price[3] > item_price[0] else 0

    def __len__(self):
        return len(self.items_indexes)

    def _pack_news_titles(self, news_titles):
        news_indexes = torch.zeros(
            (1, self.n_news, self.n_words), dtype=torch.int64
        )
        news_lengths = torch.zeros((1, self.n_news), dtype=torch.int64)
        for i, new_title in enumerate(news_titles):
            new_title_tokenized = self._clean_and_tokenize(
                new_title.decode('utf-8')
            )[:self.n_words]
            for j, new_word in enumerate(new_title_tokenized):
                news_indexes[0, i, j] = self.words_list.index(new_word) \
                    if new_word in self.words_list else 1
            news_lengths[0, i] = len(new_title_tokenized)
        return news_indexes.squeeze(0), news_lengths.squeeze(0)

    def _clean_and_tokenize(self, text):
        """Cleans the input string ``text``.

        Cleans a Reuters-specific new headline following the procedure
        explained in the report of the thesis. In short:

            1. Removes the introductory text of many Reuter headlines,
            if present.
            2. Removes possible non-informative tags at the end of the
            headline.
            3. Transforms the text to lowercase.
            4. Removes points that may be present.
            5. Removes numbers that may be present.
            6. Removes commas that may be present.
            7. Replaces special characters with spaces.
            8. Removes non-single spaces.
            9. Removes possible spaces at the beginning and the end of the
            headline (inserted by previous steps).

        Args:
            text (str): Reuters headline.

        Returns:
            str: sanitized Reuters headline.
        """
        text_sanitized = re.sub(r'^[A-Z0-9-\s]*-', '', text)
        text_sanitized = re.sub(r'\s-[\s\w]*$', '', text_sanitized)
        text_sanitized = text_sanitized.lower()
        text_sanitized = re.sub(r'[\.]+', '', text_sanitized)
        text_sanitized = re.sub(r'\w*\d\w*', '', text_sanitized)
        text_sanitized = re.sub(r'[\']', '', text_sanitized)
        text_sanitized = re.sub(r'[^a-z0-9\s]', ' ', text_sanitized)
        text_sanitized = re.sub(r'\s{2,}', ' ', text_sanitized)
        text_sanitized = re.sub(r'^\s|\s$', '', text_sanitized)
        text_sanitized = text_sanitized.translate(
            str.maketrans('', '', string.punctuation)
        )
        return self.tokenizer(text_sanitized)
