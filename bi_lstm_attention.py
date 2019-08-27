import torch
import torch.nn as nn
from torch.autograd import Variable
import const


class AttentionModel(torch.nn.Module):
    def __init__(self, args):
        super(AttentionModel, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : num classes
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_dim : Embeddding dimension of GloVe word embeddings

        --------
        """
        self.batch_size = args.batch_size
        self.output_size = args.output_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.bidirectional = args.bidirectional
        self.dropout = args.dropout
        self.use_cuda = args.cuda
        self.sequence_length = args.sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        self.lookup_table.weight.data.uniform_(-1., 1.)
        self.layer_size = args.layer_size
        self.lstm = nn.LSTM(input_size=self.embed_dim,hidden_size=self.hidden_size,num_layers=self.layer_size,
                            dropout=self.dropout,bidirectional=self.bidirectional)
        if self.bidirectional:
            self.num_direction =  2
        else:
            self.num_direction = 1

        self.attention_size = args.attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(self.hidden_size * self.num_direction, self.output_size)

    def attention_net(self, lstm_output):
        #lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        """
        print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        print(attn_tanh.size())  (squence_length * batch_size, attention_size)
        print(attn_hidden_layer.size())  (squence_length * batch_size, 1)
        print(exps.size())  (batch_size, squence_length)
        print(alphas.size()) (batch_size, squence_length)
        print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        """

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.num_direction])
        # M = tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        state = lstm_output.permute(1, 0, 2)
        # r = H*alpha.T
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            hidden_state = Variable(torch.zeros(self.layer_size*self.num_direction, len(input_sentences), self.hidden_size).cuda())
            cell_state = Variable(torch.zeros(self.layer_size*self.num_direction, len(input_sentences), self.hidden_size).cuda())
        else:
            hidden_state = Variable(torch.zeros(self.layer_size*self.num_direction, len(input_sentences), self.hidden_size))
            cell_state = Variable(torch.zeros(self.layer_size*self.num_direction, len(input_sentences), self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state,cell_state))
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits