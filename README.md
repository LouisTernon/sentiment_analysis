# Usage

```
pip install -r requirement.txt
python -m spacy download en_core_web_md
python src/tester.py
```

# Results

## Considering sentences only

Batch: 32, Epoch: 20, Optim: Adam, LR: 5e-2

| Encoder        | Lemmatization           | Pre-trained embedding  | Accuracy | Exec time| 
| ------------- |:-------------:| :-----:| :-----:|:-----:|
|*LSTM(n_emb, n_hid, n_layer)*|*bool*|*bool*|*mean (std)*|*time per run (s)*|
| LSTM(150, 64, 1)     | False | False |53.62 (3.96) |17|
| LSTM(150, 64, 1)      | True      |   False |63.08 (3.86) |85|
| LSTM(300, 64, 1) | True      |    True |74.63 (2.25)| 93|



## Concatenating one-hot encoded class before last layer

Batch: 32, Epoch: 40, Optim: Adam, LR: 5e-3

| Encoder        | Lemmatization           | Pre-trained embedding  | Accuracy | Exec time| 
| ------------- |:-------------:| :-----:| :-----:|:-----:|
|*LSTM(n_emb, n_hid, n_layer)*|*bool*|*bool*|*mean (std)*|*time per run (s)*|
| LSTM(150, 64, 1)     | False | False |55.27 (2.88) |18 (20e)|
| LSTM(300, 128, 1) | True      |    True |79.47 (0.78)| 106|