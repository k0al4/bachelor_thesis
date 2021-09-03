import argparse

import pytorch_lightning as pl

from . import BachelorThesisData, BachelorThesisModel


def main(args):
    data = BachelorThesisData(**vars(args))
    data.prepare_data()

    model = BachelorThesisModel(data.words_embeddings, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = BachelorThesisData.add_data_specific_args(parser)
    parser = BachelorThesisModel.add_model_specific_args(parser)

    main(parser.parse_args())
