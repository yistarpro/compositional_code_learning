## Compressing Word Embeddings Via Deep Compositional Code Learning (Pytorch Implementation)

This repository is modification of implementation of following: 
- https://github.com/mingu600/compositional_code_learning

It contains an implementation of the following work:

[SN18] Shu, R., Nakayama, H. (2018). [Compressing Word Embeddings Via Deep Compositional Code Learning](https://arxiv.org/pdf/1711.01068.pdf).

The purpose of this repository is to implement results from following works:

[KPLC24] Jae-yun Kim, Saerom Park, Joohee Lee, Jung Hee Cheon: Privacy-preserving embedding via look-up table evaluation with fully homomorphic encryption. Forty-first International Conference on Machine Learning, 2024.

Former implementations contains:

- compression of embedding layer via deep learning : on GloVe42B300d.
- sentimental analysis from [SN18].

Our modification added followings:

- compression of embedding layer via deep learning : on  GloVe6B50d, GloVe42B300d, GPT-2, BERT.
- simplified sentimental analysis with logistic regression for encrypted status.
- exporting learned parameters of the embeddings and the logistic regression model.

After construction of model, model is exported and the rest of the experiment continue in following CKKSEIF library, based on OpenFHE:
- https://github.com/yistarpro/CKKSEIF


## Dependencies
* Python 3
* Pytorch (version 0.4.0)
* Torchtext
* Numpy
* GloVe vectors (Download glove.42B.300d.zip from https://nlp.stanford.edu/projects/glove/)

## How to use

1. The first step is construction of the embedding for learning codes, with various embedding models.

- construct_embeedings.py
- construct_embeedings_bert.py
- construct_embeedings_gpt.py

2. The seconds step is training compressed embeddings.

- train_code_learner.py

This automatically compress all four models with various parameters, but users can specify the setting with options.

3. Using compressed embeddings, following code performs downstream task specified in [SN18]

- train_classifier.py

We also provide following experiment with HE friendly sentimental analysis.

- train_classifier_logreg.py

These codes automatically performs test on two GloVe models with various parameters, but users can specify the setting with options.

4. Save learned parameters to txt form, to use in CKKSEIF library.

- code_to_txt.py

This automatically saves all learned parameters, but users can specify the setting with options.

The result shold be moved to CKKSEIF/data, but we provide the result in following link:
- https://drive.google.com/file/d/17YVk3uR_Q25j0ebJzyrblDupi1aMtwhz/view?usp=sharing

5. misc.

- utils.py contains various utility functions.
- models.py contains the code for model.
- code_analysis.py contains various evaluation functions for checking the performance. Users can specify the setting with options. "ClassifierEvaluator" evaluates the classifier model, and "CompressionEvaluator" evaluates the compression model. 
