# DataLoader for Seq2seq
Efficient data loader for text dataset using [torch.utils.data.Dataset](https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py#L7-L36), [collate_fn](https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py#L39-L75) and [torch.utils.data.DataLoader](https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py#L97-L100).



#### Update Seq2Seq Dataloader from [yunjey/seq2seq-dataloader](<https://github.com/yunjey/seq2seq-dataloader>).

<p align="center"> <img width="500" src="https://cdn-images-1.medium.com/max/1200/1*_rSHLjFShknAu3jt3rbcNQ.png" /></p>

> Seq2Seq Model image from [Seq2Seq model in TensorFlow Post](<https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f>)



1. Add `<start>` token in decoder input, `<end>` token in target output words of model. 

```python
I am a Student => Je suis etudiant
encoder input : 'I', 'am', 'a', 'Student'
decoder input : '<start>', 'Je', 'suis', 'etudiant'
target 		  : 'Je', 'suis', 'etudiant', '<end>'
```

Please See this example.

```
Der weltweit zweitgrößte Anbieter von Besucherattraktionen zielt darauf ab , seinen 30 Millionen Besuchern auf der ganzen Welt durch seine globalen und lokalen Marken sowie das Engagement und die Leidenschaft seiner Führungskräfte und Mitarbeiter ein einzigartiges , unvergessliches und lohnenswertes Erlebnis zu bieten .
```

```python
print(trg_seqs[0])
tensor([   1,   49, 2267,    3, 4091,   68,    3, 2651,  152,  419,    8,  331,
         229,  524, 1680,  212,   49,  299, 1235,  156,  944, 3192,   14,  357,
        2454,  117,   23, 4624,   14,   50, 3648, 1819,    3,   14,  317,  171,
           3,    8,    3,   14,    3, 2676,  127, 1207,   28,    0,    0,    0,
           0])

print(target[0])
tensor([  49, 2267,    3, 4091,   68,    3, 2651,  152,  419,    8,  331,  229,
         524, 1680,  212,   49,  299, 1235,  156,  944, 3192,   14,  357, 2454,
         117,   23, 4624,   14,   50, 3648, 1819,    3,   14,  317,  171,    3,
           8,    3,   14,    3, 2676,  127, 1207,   28,    2,    0,    0,    0,
           0])
```



2. Add replace `UNK` Token Mechanism in OOV(out of vocabulary) Problem.

```python
sequence.extend([word2id[token] if token in word2id else word2id['<unk>'] for token in tokens])
```



3. Add `trg_max`, `src_max` to avoid cuda memory leak.

- `src_max` : maximum length source domain.
- `trg_max` : maximum length target domain.

This can avoid memory leak when getting high dimension of input sequence length. 


<br>


## Prerequesites
* [PyThon 2.7 or 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.1.12](http://pytorch.org/)
* [NLTK](http://www.nltk.org/)


<br>

## Usage 

#### 1. Clone the repository
```bash
$ git clone https://github.com/graykode/seq2seq-dataloader.git
$ cd seq2seq-dataloader
```

#### 2. Download nltk tokenizer
```bash
$ pip install nltk
$ python
$ import nltk
$ nltk.download('punkt')
```

#### 3. Build word2id dictionary 

```bash
$ python build_vocab.py
```

#### 4. Check DataLoader
For usage, please see [example.ipynb](https://github.com/graykode/seq2seq-dataloader/blob/master/example.ipynb).

