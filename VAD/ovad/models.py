from typing import Optional
import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import functional as F
from snowfall.models import AcousticModel
from snowfall.training.diagnostics import measure_weight_norms
class StatsPool(nn.Module):
    """Statistics pooling
    Compute temporal mean and (unbiased) standard deviation
    and returns their concatenation.
    Reference
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """




class TdnnLstm1a(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool1d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Conv1d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool1d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, affine=False),
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=512, hidden_size=512, num_layers=1) for _ in range(5)]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=512, affine=False) for _ in range(5)]
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(
        self, tb_writer: SummaryWriter, global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            "train/weight_l2_norms",
            measure_weight_norms(self, norm="l2"),
            global_step=global_step,
        )
        tb_writer.add_scalars(
            "train/weight_max_norms",
            measure_weight_norms(self, norm="linf"),
            global_step=global_step,
        )


class TdnnLstm1b(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.MaxPool1d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=60, affine=False),
            nn.Conv1d(
                in_channels=60, out_channels=60, kernel_size=3, stride=1, padding=1
            ),
            # nn.MaxPool1d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=60, affine=False),
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=60, hidden_size=60, num_layers=1) for _ in range(5)]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=60, affine=False) for _ in range(5)]
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=60, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall
        return x

    def write_tensorboard_diagnostics(
        self, tb_writer: SummaryWriter, global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            "train/weight_l2_norms",
            measure_weight_norms(self, norm="l2"),
            global_step=global_step,
        )
        tb_writer.add_scalars(
            "train/weight_max_norms",
            measure_weight_norms(self, norm="linf"),
            global_step=global_step,
        )



class TdnnLstm1c(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=60,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.MaxPool1d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=60, affine=False),
            nn.Conv1d(
                in_channels=60, out_channels=60, kernel_size=3, stride=1, padding=1
            ),
            # nn.MaxPool1d(3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=60, affine=False),
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=60, hidden_size=60, num_layers=1) for _ in range(5)]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=60, affine=False) for _ in range(5)]
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=60, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(
        self, tb_writer: SummaryWriter, global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            "train/weight_l2_norms",
            measure_weight_norms(self, norm="l2"),
            global_step=global_step,
        )
        tb_writer.add_scalars(
            "train/weight_max_norms",
            measure_weight_norms(self, norm="linf"),
            global_step=global_step,
        )


class TdnnLstm1d(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(
        self, num_features: int, num_classes: int, subsampling_factor: int = 3
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=60,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.MaxPool1d(3, stride=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=60, affine=False),
            nn.Conv1d(
                in_channels=60, out_channels=60, kernel_size=3, stride=1, padding=1
            ),
            # nn.MaxPool1d(3, stride=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=60, affine=False),
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=60, hidden_size=60, num_layers=1) for _ in range(5)]
        )
        self.lstm_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(num_features=60, affine=False) for _ in range(5)]
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features=60, out_features=self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.tdnn(x)
        x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F) -> how LSTM expects it
        for lstm, bnorm in zip(self.lstms, self.lstm_bnorms):
            x_new, _ = lstm(x)
            x_new = bnorm(x_new.permute(1, 2, 0)).permute(
                2, 0, 1
            )  # (T, B, F) -> (B, F, T) -> (T, B, F)
            x_new = self.dropout(x_new)
            x = x_new + x  # skip connections
        x = x.transpose(
            1, 0
        )  # (T, B, F) -> (B, T, F) -> linear expects "features" in the last dim
        x = self.linear(x)
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) -> shape expected by Snowfall
        # x = nn.functional.log_softmax(x, dim=1)
        return x

    def write_tensorboard_diagnostics(
        self, tb_writer: SummaryWriter, global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            "train/weight_l2_norms",
            measure_weight_norms(self, norm="l2"),
            global_step=global_step,
        )
        tb_writer.add_scalars(
            "train/weight_max_norms",
            measure_weight_norms(self, norm="linf"),
            global_step=global_step,
        )

# class Transformer(AcousticModel):
#     """
#     Args:
#         num_features (int): Number of input features
#         num_classes (int): Number of output classes
#         subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
#         d_model (int): attention dimension
#         nhead (int): number of head
#         dim_feedforward (int): feedforward dimention
#         num_encoder_layers (int): number of encoder layers
#         num_decoder_layers (int): number of decoder layers
#         dropout (float): dropout rate
#         normalize_before (bool): whether to use layer_norm before the first block.
#         vgg_frontend (bool): whether to use vgg frontend.
#     """

#     def __init__(self, num_features: int, num_classes: int, subsampling_factor: int = 4,
#                  d_model: int = 256, nhead: int = 4, dim_feedforward: int = 2048,
#                  num_encoder_layers: int = 12, num_decoder_layers: int = 6,
#                  dropout: float = 0.1, normalize_before: bool = True,
#                  vgg_frontend: bool = False, mmi_loss: bool = True) -> None:
#         super().__init__()
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.subsampling_factor = subsampling_factor
#         if subsampling_factor != 4:
#             raise NotImplementedError("Support only 'subsampling_factor=4'.")

#         self.encoder_embed = (VggSubsampling(num_features, d_model) if vgg_frontend else
#                               Conv2dSubsampling(num_features, d_model))
#         self.encoder_pos = PositionalEncoding(d_model, dropout)

#         encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, normalize_before=normalize_before)

#         if normalize_before:
#             encoder_norm = nn.LayerNorm(d_model)
#         else:
#             encoder_norm = None

#         self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

#         self.encoder_output_layer = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(d_model, num_classes)
#         )

#         if num_decoder_layers > 0:
#             if mmi_loss:
#                 self.decoder_num_class = self.num_classes + 1  # +1 for the sos/eos symbol
#             else:
#                 self.decoder_num_class = self.num_classes  # bpe model already has sos/eos symbol

#             self.decoder_embed = nn.Embedding(self.decoder_num_class, d_model)
#             self.decoder_pos = PositionalEncoding(d_model, dropout)

#             decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, normalize_before=normalize_before)

#             if normalize_before:
#                 decoder_norm = nn.LayerNorm(d_model)
#             else:
#                 decoder_norm = None

#             self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

#             self.decoder_output_layer = torch.nn.Linear(d_model, self.decoder_num_class)

#             self.decoder_criterion = LabelSmoothingLoss(self.decoder_num_class)
#         else:
#             self.decoder_criterion = None

#     def forward(self, x: Tensor, supervision: Optional[Supervisions] = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
#         """
#         Args:
#             x: Tensor of dimension (batch_size, num_features, input_length).
#             supervision: Supervison in lhotse format, get from batch['supervisions']
#         Returns:
#             Tensor: After log-softmax tensor of dimension (batch_size, number_of_classes, input_length).
#             Tensor: Before linear layer tensor of dimension (input_length, batch_size, d_model).
#             Optional[Tensor]: Mask tensor of dimension (batch_size, input_length) or None.
#         """
#         encoder_memory, memory_mask = self.encode(x, supervision)
#         x = self.encoder_output(encoder_memory)
#         return x, encoder_memory, memory_mask

#     def encode(self, x: Tensor, supervisions: Optional[Supervisions] = None) -> Tuple[Tensor, Optional[Tensor]]:
#         """
#         Args:
#             x: Tensor of dimension (batch_size, num_features, input_length).
#             supervisions : Supervison in lhotse format, i.e., batch['supervisions']
#         Returns:
#             Tensor: Predictor tensor of dimension (input_length, batch_size, d_model).
#             Optional[Tensor]: Mask tensor of dimension (batch_size, input_length) or None.
#         """
#         x = x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)

#         x = self.encoder_embed(x)
#         x = self.encoder_pos(x)
#         x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
#         mask = encoder_padding_mask(x.size(0), supervisions)
#         mask = mask.to(x.device) if mask != None else None
#         x = self.encoder(x, src_key_padding_mask=mask)  # (T, B, F)

#         return x, mask

#     def encoder_output(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor of dimension (input_length, batch_size, d_model).
#         Returns:
#             Tensor: After log-softmax tensor of dimension (batch_size, number_of_classes, input_length).
#         """
#         x = self.encoder_output_layer(x).permute(1, 2, 0)  # (T, B, F) ->(B, F, T)
#         x = nn.functional.log_softmax(x, dim=1)  # (B, F, T)
#         return x

#     def decoder_forward(self, x: Tensor, encoder_mask: Tensor, supervision: Supervisions = None,
#             graph_compiler: object = None, token_ids: List[int] = None) -> Tensor:
#         """
#         Args:
#             x: Tensor of dimension (input_length, batch_size, d_model).
#             encoder_mask: Mask tensor of dimension (batch_size, input_length)
#             supervision: Supervison in lhotse format, get from batch['supervisions']
#             graph_compiler: use graph_compiler.L_inv (Its labels are words, while its aux_labels are phones)
#                             , graph_compiler.words and graph_compiler.oov
#         Returns:
#             Tensor: Decoder loss.
#         """
#         if supervision is not None and graph_compiler is not None:
#             batch_text = get_normal_transcripts(supervision, graph_compiler.lexicon.words, graph_compiler.oov)
#             ys_in_pad, ys_out_pad = add_sos_eos(batch_text, graph_compiler.L_inv, self.decoder_num_class - 1,
#                                                 self.decoder_num_class - 1)
#         elif token_ids is not None:
#             # speical token ids:
#             # <blank> 0
#             # <UNK> 1
#             # <sos/eos> self.decoder_num_class - 1
#             sos_id = self.decoder_num_class - 1
#             eos_id = self.decoder_num_class - 1
#             _sos = torch.tensor([sos_id])
#             _eos = torch.tensor([eos_id])
#             ys_in = [torch.cat([_sos, torch.tensor(y)], dim=0) for y in token_ids]
#             ys_out = [torch.cat([torch.tensor(y), _eos], dim=0) for y in token_ids]
#             ys_in_pad = pad_list(ys_in, eos_id)
#             ys_out_pad = pad_list(ys_in, -1)

#         else:
#             raise ValueError("Invalid input for decoder self attetion")


#         ys_in_pad = ys_in_pad.to(x.device)
#         ys_out_pad = ys_out_pad.to(x.device)

#         tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(x.device)

#         tgt_key_padding_mask = decoder_padding_mask(ys_in_pad)

#         tgt = self.decoder_embed(ys_in_pad)  # (B, T) -> (B, T, F)
#         tgt = self.decoder_pos(tgt)
#         tgt = tgt.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
#         pred_pad = self.decoder(tgt=tgt,
#                                 memory=x,
#                                 tgt_mask=tgt_mask,
#                                 tgt_key_padding_mask=tgt_key_padding_mask,
#                                 memory_key_padding_mask=encoder_mask)  # (T, B, F)
#         pred_pad = pred_pad.permute(1, 0, 2)  # (T, B, F) -> (B, T, F)
#         pred_pad = self.decoder_output_layer(pred_pad)  # (B, T, F)

#         decoder_loss = self.decoder_criterion(pred_pad, ys_out_pad)

#         return decoder_loss
