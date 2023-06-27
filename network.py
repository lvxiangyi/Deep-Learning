import os
import sys
from Transformer import Transformer, TransModel
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import random_split
def get_reviews(dir_path, label):
    reviews = []
    ctr = 1
#     dir_path = os.path.join(dir_path, label)
    dir_path = '%s/%s'%(dir_path,label)
    for file in os.listdir(dir_path):
        curr_file = os.path.join(dir_path, file)
        f = open(curr_file, "r", encoding="utf8")  # one line
        for line in f:
            reviews.append((label, line))
    f.close()  # close curr file
    ctr += 1
    return reviews
dir_path='di'
label='li'
dir_path = '%s/%s'%(dir_path,label)
print(dir_path)


class IMDB(Dataset):
    def __init__(self, split='train'):
        if split == 'train':
            dir_path = r"F:\github\IMDB\aclImdb\train"
        else:
            dir_path = r"F:\github\IMDB\aclImdb\test"

        self.reviews = get_reviews(dir_path, 'pos')
        self.reviews += get_reviews(dir_path, 'neg')

    def __getitem__(self, idx):
        return self.reviews[idx]

    def __len__(self):
        return len(self.reviews)

train_dataset_raw = IMDB(split="train")
test_dataset_raw = IMDB(split="test")
print("Train dataset size: ", len(train_dataset_raw))
print("Test dataset size: ", len(test_dataset_raw))
train_dataset_raw[0]
train_set_size = 20000
valid_set_size = 5000

#train_dataset, valid_dataset = random_split(list(train_dataset_raw), [20000, 5000])
train_dataset, valid_dataset = random_split(list(train_dataset_raw)[:5000], [4000,1000])
train_dataset[0]
import re
def tokenizer(text):
    # step 1. remove HTML tags. they are not helpful in understanding the sentiments of a review
    # step 2: use lowercase for all text to keep symmetry
    # step 3: extract emoticons. keep them as they are important sentiment signals
    # step 4: remove punctuation marks
    # step 5: put back emoticons
    # step 6: generate word tokens
    text = re.sub("<[^>]*>", "", text)
    text = text.lower()
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text)
    text = text + " ".join(emoticons).replace("-", "")
    tokenized = text.split()
    return tokenized
example_text = '''This is awesome movie <br /><br />. I loved it so much :-) I\'m goona watch it again :)'''
example_text
example_tokens = tokenizer(example_text)
example_tokens

from collections import Counter
from collections import OrderedDict

token_counts = Counter()

for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

print('IMDB vocab size:', len(token_counts))

sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)


class Vocab():
    def __init__(self, ordered_dict):
        voc = {}
        voc['<pad>'] = 0
        voc['<unk>'] = 1
        for i, (k, v) in enumerate(ordered_dict.items()):
            voc[k] = i + 2
        self.voc = voc

    def __getitem__(self, key):
        if key not in self.voc:
            self.voc[key] = self.voc['<unk>']
        return self.voc[key]

    def __len__(self):
        return len(self.voc)

voc = Vocab(ordered_dict)
len(voc)

for token in ["this", "is", "an", ":)"]:
    print(token, " --> ", voc[token])

text_pipeline = lambda x: [voc[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1.0 if x == 'pos' else 0.0
text_pipeline(example_text)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print('Using device:', device)


def collate_batch(batch):
    # a function to apply pre-preprocessing steps at a batch level
    label_list, text_list, length_list = [], [], []

    # iterate over all reviews in a batch
    for _label, _text in batch:
        # label preprocessing
        label_list.append(label_pipeline(_label))
        # text preprocessing
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)

        # store the processed text in a list
        text_list.append(processed_text)

        # store the length of processed text
        # this will come handy in furture when we want to know the original size of a text
        length_list.append(processed_text.size(0))

    labels = torch.tensor(label_list)
    lengths = torch.tensor(length_list)

    # pad the processed reviews to make their lengths consistent
    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    # return
    # 1. a list of processed and padded review texts
    # 2. a list of processed labels
    # 3. a list of review text original lengths (before padding)
    return padded_texts.to(device), labels.to(device), lengths.to(device)

from torch.utils.data import DataLoader

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

text_batch, label_batch, length_batch = next(iter(dataloader))

print("text_batch.shape: ", text_batch.shape)
print("label_batch: ", label_batch)
print("length_batch: ", length_batch)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, hn = self.rnn(out)
        out = hn[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hn, cn) = self.lstm(out)
        out = hn[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
class LSTMWithAttention(nn.Module):

    # def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    #     super(LSTM, self).__init__()
    #
    #     self.hidden_dim = hidden_dim  # 隐层大小
    #     self.num_layers = num_layers  # LSTM层数
    #     # embed_dim为每个时间步对应的特征数
    #     self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2)
    #     # input_dim为特征维度，就是每个时间点对应的特征数量，这里为4
    #     self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    #     self.fc = nn.Linear(hidden_dim, output_dim)
    def __init__(self, vocab_size, embed_dim, hidden_dim, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, text,lengths,query, key, value):
        # #         print(query.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        # attention_output, attn_output_weights = self.attention(query, key, value)
        # #         print(attention_output.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        # output, (h_n, c_n) = self.lstm(attention_output)
        # #         print(output.shape) # torch.Size([16, 1, 64]) batch_size, time_step, hidden_dim
        # batch_size, timestep, hidden_dim = output.shape
        # output = output.reshape(-1, hidden_dim)
        # output = self.fc(output)
        # output = output.reshape(timestep, batch_size, -1)
        # return output[-1]
        out = self.embedding(text)
        # out = nn.utils.rnn.pack_padded_sequence(
        #     out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        # )
        attention_output, attn_output_weights = self.attention(out, out, out)
        out, (hn,cn) = self.lstm(attention_output)
        out = hn[-1, :, :]


        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class Trainer():
    def __init__(self, kwargs):
        embed_dim = kwargs['embed_dim']
        rnn_hidden_size = kwargs['rnn_hidden_size']
        fc_hidden_size = kwargs['fc_hidden_size']
        batch_size = kwargs['batch_size']

        self.num_epochs = kwargs['num_epochs']

        # Dataset
        train_dataset_raw = IMDB(split="train")
        #         test_dataset = IMDB(split="test")

        train_set_size = 20000
        valid_set_size = 5000
        train_dataset, valid_dataset = random_split(list(train_dataset_raw), [20000, 5000])

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_batch)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=collate_batch)
        #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        # Make vocabulary
        token_counts = Counter()
        for label, line in train_dataset:
            tokens = tokenizer(line)
            token_counts.update(tokens)

        sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        voc = Vocab(ordered_dict)

        # Model (modifiy here if other model is adopted)
        vocab_size = len(voc)

        self.model = SimpleRNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Criterion
        self.criterion = nn.BCELoss()

        # Opitmizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        for epoch in range(self.num_epochs):
            acc_train, loss_train = self.train_per_epoch()
            acc_valid, loss_valid = self.valid_per_epoch()
            print(
                f"Epoch {epoch} train accuracy: {acc_train:.4f}; val accuracy: {acc_valid:.4f}"
            )

    def train_per_epoch(self):
        self.model.train()
        total_acc, total_loss = 0, 0
        batch_index = 0
        train_dataset_len = len(self.train_dataloader.dataset)
        for text_batch, label_batch, lengths in self.train_dataloader:
            self.optimizer.zero_grad()
            pred = self.model(text_batch, lengths)[:, 0]
            loss = self.criterion(pred, label_batch)
            loss.backward()
            self.optimizer.step()
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
            if (batch_index + 1) % 100 == 0:
                print(f'{batch_index + 1:3d} / {train_dataset_len // self.train_dataloader.batch_size}')
            batch_index += 1
        return total_acc / train_dataset_len, total_loss / train_dataset_len

    def valid_per_epoch(self):
        self.model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for text_batch, label_batch, lengths in self.valid_dataloader:
                pred = self.model(text_batch, lengths)[:, 0]
                loss = self.criterion(pred, label_batch)
                total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)
        return total_acc / len(self.valid_dataloader.dataset), total_loss / len(self.valid_dataloader.dataset)

    def classify_review(self, text):
        text_list, lengths = [], []

        # process review text with text_pipeline
        # note: "text_pipeline" has dependency on data vocabulary
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)

        # get processed review tokens length
        lengths.append(processed_text.size(0))
        lengths = torch.tensor(lengths)

        # change the dimensions from (torch.Size([8]), torch.Size([1, 8]))
        # nn.utils.rnn.pad_sequence(text_list, batch_first=True) does this too
        padded_text_list = torch.unsqueeze(processed_text, 0)

        # move tensors to correct device
        padded_text_list = padded_text_list.to(device)
        lengths = lengths.to(device)

        # get prediction
        self.model.eval()
        pred = self.model(padded_text_list, lengths)
        print("model pred: ", pred)

        # positive or negative review
        review_class = 'negative'  # else case
        if (pred >= 0.5) == 1:
            review_class = "positive"

        print("review type: ", review_class)
class LSTMTrainer():
    def __init__(self, kwargs):
        embed_dim = kwargs['embed_dim']
        rnn_hidden_size = kwargs['rnn_hidden_size']
        fc_hidden_size = kwargs['fc_hidden_size']
        batch_size = kwargs['batch_size']

        self.num_epochs = kwargs['num_epochs']

        # Dataset
        train_dataset_raw = IMDB(split="train")
        #         test_dataset = IMDB(split="test")

        train_set_size = 20000
        valid_set_size = 5000
        train_dataset, valid_dataset = random_split(list(train_dataset_raw), [20000, 5000])

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_batch)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=collate_batch)
        #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        # Make vocabulary
        token_counts = Counter()
        for label, line in train_dataset:
            tokens = tokenizer(line)
            token_counts.update(tokens)

        sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        voc = Vocab(ordered_dict)

        # Model (modifiy here if other model is adopted)
        vocab_size = len(voc)

        self.model = LSTM(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Criterion
        self.criterion = nn.BCELoss()

        # Opitmizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        for epoch in range(self.num_epochs):
            acc_train, loss_train = self.train_per_epoch()
            acc_valid, loss_valid = self.valid_per_epoch()
            print(
                f"Epoch {epoch} train accuracy: {acc_train:.4f}; val accuracy: {acc_valid:.4f}"
            )

    def train_per_epoch(self):
        self.model.train()
        total_acc, total_loss = 0, 0
        batch_index = 0
        train_dataset_len = len(self.train_dataloader.dataset)

        for text_batch, label_batch, lengths in self.train_dataloader:
            self.optimizer.zero_grad()
            pred = self.model(text_batch, lengths)[:, 0]
            loss = self.criterion(pred, label_batch)
            loss.backward()
            self.optimizer.step()
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
            if (batch_index + 1) % 100 == 0:
                print(f'{batch_index + 1:3d} / {train_dataset_len // self.train_dataloader.batch_size}')
            batch_index += 1
        return total_acc / train_dataset_len, total_loss / train_dataset_len

    def valid_per_epoch(self):
        self.model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for text_batch, label_batch, lengths in self.valid_dataloader:
                pred = self.model(text_batch, lengths)[:, 0]
                loss = self.criterion(pred, label_batch)
                total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)
        return total_acc / len(self.valid_dataloader.dataset), total_loss / len(self.valid_dataloader.dataset)

    def classify_review(self, text):
        text_list, lengths = [], []

        # process review text with text_pipeline
        # note: "text_pipeline" has dependency on data vocabulary
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)

        # get processed review tokens length
        lengths.append(processed_text.size(0))
        lengths = torch.tensor(lengths)

        # change the dimensions from (torch.Size([8]), torch.Size([1, 8]))
        # nn.utils.rnn.pad_sequence(text_list, batch_first=True) does this too
        padded_text_list = torch.unsqueeze(processed_text, 0)

        # move tensors to correct device
        padded_text_list = padded_text_list.to(device)
        lengths = lengths.to(device)

        # get prediction
        self.model.eval()
        pred = self.model(padded_text_list, lengths)
        print("model pred: ", pred)

        # positive or negative review
        review_class = 'negative'  # else case
        if (pred >= 0.5) == 1:
            review_class = "positive"

        print("review type: ", review_class)
class LSTMWithAttentionTrainer:
    def __init__(self, kwargs):
        embed_dim = kwargs['embed_dim']
        rnn_hidden_size = kwargs['rnn_hidden_size']
        fc_hidden_size = kwargs['fc_hidden_size']
        batch_size = kwargs['batch_size']

        self.num_epochs = kwargs['num_epochs']

        # Dataset
        train_dataset_raw = IMDB(split="train")
        #         test_dataset = IMDB(split="test")

        train_set_size = 20000
        valid_set_size = 5000
        train_dataset, valid_dataset = random_split(list(train_dataset_raw), [20000, 5000])

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_batch)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=collate_batch)
        #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        # Make vocabulary
        token_counts = Counter()
        for label, line in train_dataset:
            tokens = tokenizer(line)
            token_counts.update(tokens)

        sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        voc = Vocab(ordered_dict)

        # Model (modifiy here if other model is adopted)
        vocab_size = len(voc)

        self.model = LSTMWithAttention(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Criterion
        self.criterion = nn.BCELoss()

        # Opitmizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        for epoch in range(self.num_epochs):
            acc_train, loss_train = self.train_per_epoch()
            acc_valid, loss_valid = self.valid_per_epoch()
            print(
                f"Epoch {epoch} train accuracy: {acc_train:.4f}; val accuracy: {acc_valid:.4f}"
            )

    def train_per_epoch(self):
        self.model.train()
        total_acc, total_loss = 0, 0
        batch_index = 0
        train_dataset_len = len(self.train_dataloader.dataset)

        for text_batch, label_batch, lengths in self.train_dataloader:
            self.optimizer.zero_grad()
            pred = self.model(text_batch, lengths,text_batch,text_batch,text_batch)[:, 0]
            loss = self.criterion(pred, label_batch)
            loss.backward()
            self.optimizer.step()
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
            if (batch_index + 1) % 100 == 0:
                print(f'{batch_index + 1:3d} / {train_dataset_len // self.train_dataloader.batch_size}')
            batch_index += 1
        return total_acc / train_dataset_len, total_loss / train_dataset_len

    def valid_per_epoch(self):
        self.model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for text_batch, label_batch, lengths in self.valid_dataloader:
                pred = self.model(text_batch, lengths,text_batch,text_batch,text_batch)[:, 0]
                loss = self.criterion(pred, label_batch)
                total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)
        return total_acc / len(self.valid_dataloader.dataset), total_loss / len(self.valid_dataloader.dataset)


    def classify_review(self, text):
        text_list, lengths = [], []

        # process review text with text_pipeline
        # note: "text_pipeline" has dependency on data vocabulary
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)

        # get processed review tokens length
        lengths.append(processed_text.size(0))
        lengths = torch.tensor(lengths)

        # change the dimensions from (torch.Size([8]), torch.Size([1, 8]))
        # nn.utils.rnn.pad_sequence(text_list, batch_first=True) does this too
        padded_text_list = torch.unsqueeze(processed_text, 0)

        # move tensors to correct device
        padded_text_list = padded_text_list.to(device)
        lengths = lengths.to(device)

        # get prediction
        self.model.eval()
        pred = self.model(padded_text_list, lengths)
        print("model pred: ", pred)

        # positive or negative review
        review_class = 'negative'  # else case
        if (pred >= 0.5) == 1:
            review_class = "positive"

        print("review type: ", review_class)
class TransformerTrainer:
    def __init__(self, kwargs):
        embed_dim = kwargs['embed_dim']
        nhead = kwargs['nhead']
        fc_hidden_size = kwargs['fc_hidden_size']
        batch_size = kwargs['batch_size']

        self.num_epochs = kwargs['num_epochs']

        # Dataset
        train_dataset_raw = IMDB(split="train")
        #         test_dataset = IMDB(split="test")

        train_set_size = 20000
        valid_set_size = 5000
        train_dataset, valid_dataset = random_split(list(train_dataset_raw), [20000, 5000])

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_batch)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                           collate_fn=collate_batch)
        #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        # Make vocabulary
        token_counts = Counter()
        for label, line in train_dataset:
            tokens = tokenizer(line)
            token_counts.update(tokens)

        sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        voc = Vocab(ordered_dict)

        # Model (modifiy here if other model is adopted)
        vocab_size = len(voc)

        self.model = TransModel(vocab_size, embed_dim,10, nhead, 0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Criterion
        self.criterion = nn.BCELoss()

        # Opitmizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        for epoch in range(self.num_epochs):
            acc_train, loss_train = self.train_per_epoch()
            acc_valid, loss_valid = self.valid_per_epoch()
            print(
                f"Epoch {epoch} train accuracy: {acc_train:.4f}; val accuracy: {acc_valid:.4f}"
            )

    def train_per_epoch(self):
        self.model.train()
        total_acc, total_loss = 0, 0
        batch_index = 0
        train_dataset_len = len(self.train_dataloader.dataset)

        for text_batch, label_batch, lengths in self.train_dataloader:
            self.optimizer.zero_grad()
            pred = self.model(text_batch, lengths)[:, 0]
            loss = self.criterion(pred, label_batch)
            loss.backward()
            self.optimizer.step()
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
            if (batch_index + 1) % 100 == 0:
                print(f'{batch_index + 1:3d} / {train_dataset_len // self.train_dataloader.batch_size}')
            batch_index += 1
        return total_acc / train_dataset_len, total_loss / train_dataset_len

    def valid_per_epoch(self):
        self.model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for text_batch, label_batch, lengths in self.valid_dataloader:
                pred = self.model(text_batch, lengths)[:, 0]
                loss = self.criterion(pred, label_batch)
                total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)
        return total_acc / len(self.valid_dataloader.dataset), total_loss / len(self.valid_dataloader.dataset)

    def classify_review(self, text):
        text_list, lengths = [], []

        # process review text with text_pipeline
        # note: "text_pipeline" has dependency on data vocabulary
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)

        # get processed review tokens length
        lengths.append(processed_text.size(0))
        lengths = torch.tensor(lengths)

        # change the dimensions from (torch.Size([8]), torch.Size([1, 8]))
        # nn.utils.rnn.pad_sequence(text_list, batch_first=True) does this too
        padded_text_list = torch.unsqueeze(processed_text, 0)

        # move tensors to correct device
        padded_text_list = padded_text_list.to(device)
        lengths = lengths.to(device)

        # get prediction
        self.model.eval()
        pred = self.model(padded_text_list, lengths)
        print("model pred: ", pred)

        # positive or negative review
        review_class = 'negative'  # else case
        if (pred >= 0.5) == 1:
            review_class = "positive"

        print("review type: ", review_class)




#train srnn
kwargs = {
    'num_epochs': 8,
    'batch_size': 32,
    'embed_dim': 20,
    'rnn_hidden_size': 64,
    'fc_hidden_size': 64
}
kwargs
# trainer = Trainer(kwargs)
# trainer.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #train LSTM
# LSTMTrainer = LSTMTrainer(kwargs)
# LSTMTrainer.train()

#train LSTMWithAttention
# LSTMWithAttentionTrainer = LSTMWithAttentionTrainer(kwargs)
# LSTMWithAttentionTrainer.train()

#train Transformer
kwargsTransformer = {
    'num_epochs': 8,
    'batch_size': 32,
    'embed_dim': 20,
    'nhead': 2,
    'fc_hidden_size': 64
}
#
TransformerTrainer = TransformerTrainer(kwargsTransformer)
TransformerTrainer.train()

