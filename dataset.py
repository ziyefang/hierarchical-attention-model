import re
import pandas as pd
from torch.utils.data import Dataset
import torch
from vocabulary import Vocabulary


class TextDataset(Dataset):
    """
       Loads a list of sentences into memory from a text file,
       split by newlines.
    """

    def __init__(self, input_files, label_file, vocab_file, word_limit=50, sentence_limit=200):
        self.data = []
        self.labels = []
        self.vocab_word_to_idx = Vocabulary(vocab_file).word_to_idx
        label_data = pd.read_csv(label_file)
        label_data = label_data.fillna(0)
        self.word_limit = word_limit
        self.sentence_limit = sentence_limit
        for inputfile in input_files:
            patient_name = inputfile.split(".")[0]
            mortality_data = label_data.dead_after_disch_date[label_data.patient_id == patient_name]
            mortality = mortality_data.iloc[0]
            if mortality == -1:
                self.labels.append("1")
            else:
                self.labels.append("0")
        for input_file in input_files:
            with open(input_file, 'r') as f:
                sentences = []
                sentence = []
                for line in f:
                    line = line.strip()
                    if line:
                        token = re.split(" ", line)
                        if len(token) > 1:
                            continue
                        else:
                            token = ' '.join(token)
                            sentence.append(token)
                    else:
                        sentence_join = ' '.join(sentence)
                        sentences.append(sentence_join)
                        sentence = []
                f.close()
            self.data.append(sentences)

    def transform(self, text):
        # encode document
        doc = [[self.vocab_word_to_idx[word] if word in self.vocab_word_to_idx.keys() else self.vocab_word_to_idx["<unk>"] for word in sent.split(" ")]
            for sent in text]  # if len(sent) > 0
        doc = [sent[:self.word_limit] for sent in doc][:self.sentence_limit]
        num_sents = min(len(doc), self.sentence_limit)
        num_words = [min(len(sent), self.word_limit) for sent in doc]

        # skip erroneous ones
        if num_sents == 0:
            return None, -1, None

        return doc, num_sents, num_words

    def __getitem__(self, i):
        label = self.labels[i]
        text = self.data[i]

        # NOTE MODIFICATION (REFACTOR)
        doc, num_sents, num_words = self.transform(text)

        if num_sents == -1:
            return None
        return doc, label, num_sents, num_words

    def __len__(self):
        return len(self.data)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def num_classes(self):
        return 4
        # return len(list(self.data.target_names))


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, doc_lengths, sent_lengths = list(zip(*batch))
    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])
    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()
    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.LongTensor(sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length] = torch.LongTensor(sent)
    labels_array = []
    for label in labels:
        labels_array.append(int(label))
    labels_tensor = torch.LongTensor(labels_array)
    return docs_tensor, labels_tensor, torch.LongTensor(doc_lengths), sent_lengths_tensor
