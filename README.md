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
## Dataset Copyright
Our datasets are open sourced for commercial and academic use and licensed under CC-BY-SA 4.0 license.
