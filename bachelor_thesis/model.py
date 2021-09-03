import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F


class BachelorThesisModel(pl.LightningModule):

    def __init__(self, embds_tensor, **kwargs):
        super(BachelorThesisModel, self).__init__()
        self.embd_layer = torch.nn.Embedding.from_pretrained(embds_tensor, 0)
        self.lstm_layer = torch.nn.LSTM(
            embds_tensor.shape[1], kwargs['lstm_hidden'], batch_first=True
        )
        self.linear_layer = torch.nn.Linear(kwargs['lstm_hidden'], 2)
        self.dropout_prob = kwargs['dropout_prob']
        self.save_hyperparameters('lstm_hidden', 'dropout_prob')

    def forward(self, news_indexes, news_lengths):
        x = self.embd_layer(news_indexes)

        # Apply the LSTM encoder to each new, extract the encoding at the
        # position of the last word
        news_encoded = []
        for i in range(x.size(1)):
            new_encoded, _ = self.lstm_layer(x[:, i])
            new_encoded_batch = []
            for b, new_length in enumerate(news_lengths[:, i]):
                new_encoded_batch.append(new_encoded[b, new_length - 1])
            news_encoded.append(torch.stack(new_encoded_batch))
        news_encoded = torch.stack(news_encoded, dim=1)

        # Apply Dropout and Max-Pooling over the hidden features
        news_encoded = F.dropout(news_encoded, p=self.dropout_prob)
        news_encoded_max, _ = news_encoded.max(dim=1)

        # Apply the linear layer
        return self.linear_layer(news_encoded_max)

    def training_step(self, batch, batch_idx):
        (news_indexes, news_lengths), prediction_target = batch
        prediction = self(news_indexes, news_lengths)
        return F.cross_entropy(prediction, prediction_target)

    def validation_step(self, batch, batch_idx):
        (news_indexes, news_lengths), prediction_target = batch
        prediction = self(news_indexes, news_lengths)
        return F.cross_entropy(prediction, prediction_target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('BachelorThesisModel')
        parser.add_argument('--lstm_hidden', type=int, default=1024)
        parser.add_argument('--dropout_prob', type=float, default=0.5)
        return parent_parser
