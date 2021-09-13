# TransferLearning_NP
A large proportion of lead compounds are derived from natural products. However, most natural products have not been fully tested for their targets. To help resolve this problem, a model using transfer learning was built to predict targets for natural products. The target prediction model can be applied in the field of natural product-based drug discovery and has the potential to find more lead compounds or to assist researchers in drug repurposing.
This repository contains the code to reproduce the results from our published paper ['Target Prediction Model for Natural Products Using Transfer Learning'](https://www.mdpi.com/1422-0067/22/9/4632). Only acadamic or non-commercial usage is allowed.
![image](https://github.com/qiangbo1222/TransLearn_NP/blob/main/images/Fig10.png)

## Data

The bioactivity data used for training can be derived from the offical website of [ChEMBL](https://www.ebi.ac.uk/chembl/) and the structures of natural products can be downloaded from [COCONUT](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00478-9). The code needed for cleaning and processing data are provided.

## Models
The model was pre-trained on a processed ChEMBL dataset and then fine-tuned on a natural product dataset. Benefitting from transfer learning and the data balancing technique, the model achieved a highly promising area under the receiver operating characteristic curve (AUROC) score of 0.910, with limited task-related training samples. The boost effect of model's AUROC can be viewed in the belowed Figure.
![image](https://github.com/qiangbo1222/TransLearn_NP/blob/main/images/Fig3.png)

All the model's defination can be found in pretrain.py and finetune.py


## Contributing
Bo Qiang, School of Pharmaceutical Sciences, Peking University


