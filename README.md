# MPAD-DocumentUnderstanding
This repository contains implementation of paper "Message Passing Attention Networks for Document Understanding" as part of my coursewrok BITS F312 - Neural Networks and Fuzzy Logic at BITS Pilani.

## Implementation Expectation
  1. Reproduce results of MPAD and any one of its variants on one topic modelling dataset and one binary sentiment analysis dataset and one multi-class sentiment analysis dataset. (total 2 models * 3 datasets) = 6 results to be reproduced. [You are free to pick these 3 datasets from the given 10 and also one variant of MPAD from the given 3]
  2. Pick any 2 of the ablation studies and reproduce results for each of the ablation studies on any 1 of the datasets.
  3. In all above reports, in addition to accuracy, report F1 scores, precision and recall.
  4. Plot loss, accuracy and f1 during training


The three datasets selected are:
  1. Binary Sentiment Analysis: Polarity rt-polarity.txt
  2. Multi-class Sentiment Analysis: SST1
  tab_sst1_train_n_test.txt: Contains both test and train data
  (Both training and testing are kept here as per our requirement for
  graph representation to be done at same time)
  tab_sst1_test.txt: Contains test data
  3. Topic Modelling Dataset: TREC
  tab_trec_train_n_test: Contains both test and train data
  (Both training and testing are kept here as per our requirement for
  graph representation to be done at same time)
  tab_trec_test: Contains test data

Drive link to datasets is [here](https://drive.google.com/drive/folders/1x8ZKWl3JQl687d5Zg3Lf4Kfi-IRgkaPA?usp=sharing)


## File Structure
1. MPAD(mpad folder contains the details regarding this).
Markup:
  ● main_cross_val.py => Main function using k-fold cross validation for final evaluation. This is used for Polarity dataset only.
  ● main_test.py => Main function using testing datasets for final evaluation.
  ● utils.py => Contains all utility functions. Graphical representation functions are also present here.
  ● mlp.py => Pytorch model for Multi-layer perceptron used in AGGREGATE phase.
  ● layers.py => Pytorch model for attention mechanism and Message passing mechanism (to be used in models.py)
  ● models.py => Main Pytorch model used for training & testing.

2. Hierarchical MPAD(hierarchial_mpad folder contains the details
regarding this).
Markup:
  ● hmain_cross_val.py => Main function using k-fold cross validation for final evaluation. This is used for Polarity dataset only.
  ● hmain_test.py => Main function using testing datasets for final evaluation.
  ● hutils.py => Contains all utility functions. Graphical representation functions are also present here. This is to be used for hmain_cross_val.py. Utility functions for main cross validation and test differ slightly.
  ● hutils_test.py => Contains all utility functions. Graphical representation functions are also present here. This is to be used for hmain_test.py.
  ● mlp.py => Pytorch model for Multi-layer perceptron used in AGGREGATE phase.
  ● layers.py => Pytorch model for attention mechanism and Message passing mechanism (to be used in models.py)
  ● models.py => Main Pytorch model used for training & testing.
  
  
## Execution
  Execution Commands can be found [here](https://colab.research.google.com/drive/1WxridjRsmlrwULAXRF7HXBSHd6l4EjKK?usp=sharing) (also in colab file provided,
  Execution.ipynb )
  


## REFERENCES
Markup:
  * [Message Passing Attention Networks for Document Understanding](https://arxiv.org/pdf/1908.06267.pdf) (Paper)
  * Giannis Nikolentzos, MPAD [GitHub Repository](https://github.com/giannisnik/mpad)
