### Notice

**Our dataset is currently being used for an international big data competition (https://www.ikcest.org/bigdata2023). You can find the complete dataset, along with guidance, at the following link: https://aistudio.baidu.com/datasetdetail/230144. The competition provides both training and validation sets which can aid in training your models. Please note that the full test set will be made available after the competition concludes.**

## Installation

For training, a GPU is recommended to accelerate the training speed.

### PyTroch

The code is based on PyTorch 1.6+. You can find tutorials [here](https://pytorch.org/tutorials/).

### Dependencies

The code is written in Python 3.7.6. Its dependencies are summarized in the file ```requirements.txt```. 

```
transformers==4.8.1
torchvision==0.8.2
torch==1.7.1
scikit-learn==1.0
```

You can install these dependencies like this:

```
pip3 install -r requirements.txt
```

## Usage

* Run the full model on weibo dataset with default hyperparameter settings

  - bert method： 

    ~~~
    cd training_and_evaluation
    sh run_bert.sh
    ~~~

  - sentence transformer method:

    ~~~bash
    cd training_and_evaluation
    sh run_sent.sh
    ~~~


## Data

### Format

Each dataset is a folder under the ```./queries_dataset``` folder:

```
./data
└── merge
    ├── img              #org img
    ├── img_html         #result of direct search
    ├── inverse_search   #resulf of inverse search
    ├── img_news  
    ├── img_html_news
    ├── inverse_search_news  
    ├── img_random
    ├── img_html_random
    └── inverse_search_random 

```
`WEIBO`:https://drive.google.com/file/d/1tDQ5_DWQH7cyQ84fVTYiXEcr5va4_MIB/view?usp=drive_link

`TWITTER`:https://drive.google.com/file/d/1-2_DhQWx8Cyh2MvtZeFkdfh31tuzTUDD/view?usp=drive_link

## Dataset Copyright

Our datasets are open sourced for commercial and academic use and licensed under CC-BY-SA 4.0 license.
