# Usage

```
pip install -r requirement.txt
python -m spacy download en_core_web_md
python src/tester.py
```

# Results

## Considering sentences only

| Encoder        | Lemmatization           | Pre-trained embedding  | Accuracy | Exec time| 
| ------------- |:-------------:| :-----:| :-----:|:-----:|
|*LSTM(n_emb, n_hid, n_layer)*|*bool*|*bool*|*mean (std)*|*time per run (s)*|
| LSTM(150, 64, 1)     | False | False |53.62 (3.96) |17|
| LSTM(150, 64, 1)      | True      |   False |63.08 (3.86) |85|
| LSTM(300, 64, 1) | True      |    True |74.63 (2.25)| 93|

