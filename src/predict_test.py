# coding=utf-8
import torch
import pickle
from torch.autograd import Variable

from loader import *
from utils import *

model_path = os.path.join('models', 'lct_ent', 'lct_ent_60')
mapping_path = os.path.join('models', 'lct_ent', 'mapping_lct_ent.pkl')

with open(mapping_path, 'rb') as f:
    mappings = pickle.load(f)
word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']
lower = True
use_gpu = False

model = torch.load(model_path, map_location=torch.device('cpu'))

if use_gpu:
    model.cuda()
model.eval()


def predict(text, word_to_id, char_to_id, lower=True):
    def f(x): return x.lower() if lower else x
    predicted = []
    sents = text.split('\n')
    for sent in sents:
        str_words = [w for w in sent.strip().split(' ')]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>'] for w in str_words]
        chars2 = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w] for w in str_words]
        caps = [cap_feature(w) for w in str_words]

        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(words))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            _, predicted_ids = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(),chars2_length, d)
        else:
            _, predicted_ids = model(dwords, chars2_mask, dcaps, chars2_length, d)
        for (word, pred_id) in zip(str_words, predicted_ids):
            line = ' '.join([word, id_to_tag[pred_id]])
            predicted.append(line)
        predicted.append('')

    return predicted


test = '- At least 10 years old with history of atrial fibrillation\n - Seen in ED in past 6 months'
predicted = predict(test, word_to_id, char_to_id)
print(predicted)