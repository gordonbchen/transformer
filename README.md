# transformer
Exploring how transformers work.
* Masked multi-head attention
* Sin position encoding
* Byte-pair encoding
* Transformer encoder and decoder layers
* Translator and GPT models

## Usage
* Requirements: python3, pytorch, torchinfo (for displaying model params), (matplotlib + jupyter optional for notebooks)

BPE training: Run `python3 prep_(gpt|translator)_data.py`. This will train a byte-pair encoder and save tokenized outputs (`tokens.pt`) and the trained BPE (`bpe.model`) in the `data/(dataset name)` dir.

Model training: Run `python3 train_(gpt|translator).py`. This will train a gpt or translator transformer. **BPE training must be done beforehand**. Model outputs, including a loss plot (`loss_plot.png`), new text (`new_(text|translations).txt`), and model weights (`weights.pt`) will be saved in the `outputs/(model_name)` dir.

BPE models are saved in the following format, each on a separate line: `\# of special tokens`, `special tokens`, `byte pair merges`

`models`: gpt and translator models, as well as transformer building blocks.

`datasets`: raw text datasets. Tokenized datasets and trained BPEs are contained in the `data` dir.

`todo.md`: Things I should probably do.

`notebooks`: Development notebooks with simple illustrations of topics (sin position encoding, BPE, attention). Notebooks are soley for illustration and visualization, and are **not kept up to date**. 

## Sources

### Transformer
* Andrej Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy
* Arjun Sarkar: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
* Attention Is All You Need: https://arxiv.org/pdf/1706.03762

### Byte-Pair Encoding
* Andrej Karpathy: https://www.youtube.com/watch?v=zduSFxRajkE&ab_channel=AndrejKarpathy
* Andrej Karpathy: https://github.com/karpathy/minbpe/tree/master
* Wikipedia: https://en.wikipedia.org/wiki/Byte_pair_encoding

### Neural Machine Translation
* Keras: https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
