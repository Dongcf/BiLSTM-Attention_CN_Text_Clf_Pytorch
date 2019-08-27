import re
from torchtext import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)

def _clean_data(sent, sw=None, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
        sent = re.sub('！{2,}', '！', sent)
        sent = re.sub('？{2,}', '！', sent)
        sent = re.sub('。{2,}', '。', sent)
        sent = re.sub('，{2,}', '，', sent)
        sent = re.sub('\s{2,}', ' ', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent


def word_cut(text):
    text = _clean_data(text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev,test = data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train='train.csv', validation='dev.csv',test='test.csv',
        fields=[
            ('index', None),
            ('cluster_name', label_field),
            ('text', text_field)
        ]
    )
    return train, dev,test