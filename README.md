# Bachelor Thesis
[![](https://img.shields.io/badge/publication-UPC%20Commons-red)](https://upcommons.upc.edu/handle/2117/128164)
[![](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![](https://www.codefactor.io/repository/github/davidalvarezdlt/bachelor_thesis/badge)](https://www.codefactor.io/repository/github/davidalvarezdlt/bachelor_thesis)
[![](https://img.shields.io/github/license/davidalvarezdlt/bachelor_thesis)](https://github.com/davidalvarezdlt/bachelor_thesis/blob/main/LICENSE)

This repository contains a partial implementation of my bachelor's thesis ["Real-time stock predictions with Deep Learning and news scrapping"](https://upcommons.upc.edu/handle/2117/128164).
While the data and the cleaning pipeline are exactly as described in the
report, the models and the results are not shared. Instead, a toy model for
prototyping is included.

## Preparing the Data

The data used in the thesis has been completely crawled and put together from
scratch. Specifically, you can find the titles and descriptions of the news
published on [reuters.com](https://www.reuters.com) from January 2010 to May
2018. In addition to that, you also have the stock prices (end of the day) of
2019. S&P 500 companies extracted from [alphavantage.co](https://www.alphavantage.co).

Everything is compressed in a H5DF file that you can download from
[this link](https://www.kaggle.com/davidalvarezdlt/bachelor-thesis) (3.93 GB).

In order to access the data, you must load it using ``h5py``. You can then get
the news of a certain date or the stock price movements of one of the symbols
as:

```
data = h5py.File('path/to/bachelor_thesis_data.hdf5', 'r')
date_news = data['news/reuters']['2010-01-20'][()]
stock_prices = data['prices/AAPL']['2010-01-20'][()]
```

For the case of the news, ``date_news`` is a ``np.ndarray`` of size
``(n_news, 5)`` containing the title, description, category, URL and UTC
publishing datetime of the news of that specific date.

For the case of the stock prices, ``stock_prices`` is also a ``np.ndarray`` of
size ``(8,)`` containing the opening price, maximum price, minimum price,
closing price, volume of traded stocks, dividend, and split coefficient.

Notice that not every date is available, both in the case of the news and the
stock prices. Read the documentation of HDF5 to learn more about how to deal
with this type of file.

## Running the Code

The first step is to clone this repository in your computer and install its
dependencies:

```
git clone https://github.com/davidalvarezdlt/bachelor_thesis.git
cd bachelor_thesis
pip install -r requirements.txt
```

After downloading the repository to your personal computer, make sure to move
the data to ``./bachelor_thesis/data/``. You will also have to download
[Word2Vec](https://www.kaggle.com/davidalvarezdlt/bachelor-thesis)
and store it in the same path, as it's used to get the word embeddings.

The implementation of this repository is done using [PyTorch Lightning](https://www.pytorchlightning.ai/).
Read its documentation to get a complete overview of how is this repository
organized. You can run the code using default parameters by calling:

```
python -m bachelor_thesis
```

The first time you run the code, the ``ligthning_logs/data.ckpt`` file will
be created. This process might take some minutes to complete. You can
obtain a complete list of available arguments by calling:

```
python -m bachelor_thesis --help
```

For instance, you can run the code in a GPU by calling:

```
python -m bachelor_thesis --gpus 1
```

Every time you run the code, a new version folder will be created inside
``lightning_logs``. Read the documentation of PyTorch Lightning to know
more about how to modify the default behavior of the framework.

## Citation

If you use the data provided in this repository or if you find this thesis
useful, please use the following citation:

```
@thesis{Alvarez2018,
    type = {Bachelor's Thesis},
    author = {David Álvarez de la Torre},
    title = {Real-time stock predictions with Deep Learning and news scrapping},
    school = {Universitat Politècnica de Catalunya},
    year = 2018,
}
```
