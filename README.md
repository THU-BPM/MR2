## Obtain the Dataset

- Our dataset is currently being used for an international big data competition (https://www.ikcest.org/bigdata2023). You can find the complete dataset, along with guidance, at the following link: https://aistudio.baidu.com/datasetdetail/230144. The competition provides both training and validation sets which can aid in training your models. ~~Please note that the full test set will be made available after the competition concludes.~~
- Update: We have release our full dataset in https://drive.google.com/file/d/14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4/view?usp=sharing .

## Usage

You can find the baselines and guidance at https://aistudio.baidu.com/datasetdetail/230144.

Notebook baseline: https://aistudio.baidu.com/projectdetail/6371316?sUid=2739660&shared=1&ts=1689818284921
baseline: https://aistudio.baidu.com/clusterprojectdetail/6549711

## Data

### Organization Structure

The organization structure of the dataset is as follows
```
.
├── dataset_items_test.json
├── dataset_items_train.json
├── dataset_items_val.json
├── test
├── train
└── val

```
We have merged the English and Chinese datasets together and you can split them if necessary

Labels 0, 1, 2 represent non-rumor, rumor, and unverified, respectively.
### File Format

Images for each claim are stored in the `img` folder.

The `img_html_news` folder contains web pages and images retrieved based on the caption of each claim. The folder includes a `direct_annotation.json` file containing the following information:

```json
{
      "img_link": "Link to the retrieved related image",
      "page_link": "Link to the retrieved web page",
      "domain": "Domain of the retrieved web page",
      "snippet": "Brief summary of the retrieved web page",
      "image_path": "Path to the retrieved image",
      "html_path": "Path to the retrieved web page",
      "page_title": "Title of the retrieved web page"
}
```

The `inverse_search` folder contains web pages found based on the images in the claims. The folder includes an `inverse_annotation.json` file containing the following information:

```json
{
"entities": "Entities in the image of the claim", 
"entities_scores": "Scores of the entities in the image of the claim", 
"best_guess_lbl": "The most likely description of the image in the claim", 
"all_fully_matched_captions": "", 
"all_partially_matched_captions": "",
"fully_matched_no_text": "",
//The above three fields are values of the found web pages, which are a list. Each element in the list is a dictionary, formatted as follows:
	{
	"page_link": "Link to the retrieved web page", 
	"image_link": "Link to the retrieved image", 
	"html_path": "Path to the retrieved web page", 
	"title": "Title of the retrieved web page"
	}
}
```
## Dataset Copyright

Our datasets are open sourced for commercial and academic use and licensed under CC-BY-SA 4.0 license.
