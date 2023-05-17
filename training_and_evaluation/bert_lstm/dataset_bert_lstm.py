import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import cv2 
import imghdr
from unidecode import unidecode

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    input_str = unidecode(input_str)  
    return input_str
    
class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, visual_news_root_dir, queries_root_dir, news_clip_root_dir, domain_to_idx_dict, split, sent_emb_dim, load_clip_for_queries=False, binary_feature_ent=False, binary_feature_cap=False, filter_duplicates=False,labels_overlap=False ):
        self.context_data_items_dict = context_data_items_dict
        #self.visual_news_root_dir = visual_news_root_dir
        self.queries_root_dir = queries_root_dir
        #self.news_clip_root_dir = news_clip_root_dir
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.domain_to_idx_dict = domain_to_idx_dict
        #self.visual_news_data_dict = json.load(open(os.path.join(self.visual_news_root_dir+"data.json")))
        #self.visual_news_data_mapping = {ann["id"]: ann for ann in self.visual_news_data_dict}
        #self.news_clip_data_dict = json.load(open(os.path.join(self.news_clip_root_dir,split+".json")))["annotations"]
        self.sent_emb_dim = sent_emb_dim        
        #self.load_clip_for_queries = load_clip_for_queries #whether to load precomputed clip embeddings for queries only 
        self.binary_feature_cap = binary_feature_cap
        self.binary_feature_ent = binary_feature_ent
        #self.filter_duplicates = filter_duplicates
        self.labels_overlap = labels_overlap 

    def __len__(self):
        return len(self.context_data_items_dict)   

    #load entities as text 
    def load_entities(self,inv_dict,ent_binary_path):
        entities = []
        if 'entities' in inv_dict.keys():
            entities = inv_dict['entities']
        if 'best_guess_lbl' in inv_dict.keys():
            entities = entities + inv_dict['best_guess_lbl']
        for i in range(0,len(entities)):
            entities[i] = process_string(entities[i])
        if entities:
            ent_binary_file = np.load(ent_binary_path)
            ent_binary = torch.from_numpy(ent_binary_file['arr_0'])
        else:
            ent_binary = torch.empty( (0) )    
        return entities,ent_binary 
    
    def get_domain_idx(self,domain):
        if domain in self.domain_to_idx_dict.keys(): 
            domain_idx = self.domain_to_idx_dict[domain]
        else:
           # domain_idx = self.domain_to_idx_dict['UNK']
           domain_idx = 0
        return domain_idx
            
    def load_imgs_direct_search(self,imgs_file_path, metadata_file_path, imgs_to_keep_path, labels_overlap_file, label,places=False): 
        img_dim = (2048+1) if self.labels_overlap else 2048
        # if self.filter_duplicates and label==0:
        #     imgs_to_keep_idx = json.load(open(imgs_to_keep_path))['index_of_images_tokeep']
        #     if len(imgs_to_keep_idx) == 0:
        #         imgs = torch.empty((0,img_dim)) if places else torch.empty((0,img_dim,7,7)) 
        #         domains = torch.empty((0))
        #         if places: return imgs
        #         return imgs, domains
        #     imgs_to_keep_idx = np.asarray([int(i) for i in imgs_to_keep_idx])   
        imgs = np.load(imgs_file_path)['arr_0']
        # if self.filter_duplicates and label==0: imgs = imgs[imgs_to_keep_idx,:]
        imgs = torch.from_numpy(imgs)
        if self.labels_overlap: 
            labels_overlap_feature = np.load(labels_overlap_file)['arr_0']
            # if self.filter_duplicates and label==0: labels_overlap_feature=labels_overlap_feature[imgs_to_keep_idx]
            labels_overlap_feature = torch.from_numpy(labels_overlap_feature).float()
            if places: 
                imgs = torch.cat( (imgs,labels_overlap_feature.unsqueeze(1)), dim=1)             
            else:
                labels_overlap_feature = labels_overlap_feature.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                labels_overlap_feature = labels_overlap_feature.expand(-1,-1,7,7)
                imgs = torch.cat((imgs,labels_overlap_feature),dim=1)   
        #we already read the domains before                 
        if places: return imgs                 
        imgs_metadata = json.load(open(metadata_file_path)) 
        domains = []
        for i in range(0, len(imgs_metadata)):
            # if self.filter_duplicates and label==0: 
            #     if i not in imgs_to_keep_idx: continue            
            domain_idx = self.get_domain_idx(imgs_metadata[str(i)]['domain'])
            domains.append(domain_idx)               
        domains = torch.as_tensor(domains)
        return imgs, domains

    def load_queries(self,idx, qimg_path, qcap_path=None, visual_news_caption_item=None, qplace_path=None):
        #load caption as text 
        # if visual_news_caption_item: 
        #     caption = visual_news_caption_item['caption']
        # else:
        key = self.idx_to_keys[idx]
        caption = self.context_data_items_dict[key]['caption']
        img_file = np.load(qimg_path)
        img = torch.from_numpy(img_file['arr_0'])
        if qplace_path:
            places_file = np.load(qplace_path)
            qplaces = torch.from_numpy(places_file['arr_0'])
            return caption, img, qplaces
        return caption, img

    # def load_captions(self, captions_info_path, captions_binary_path, captions_to_keep_path):
    #     #filter exact matches by default..
    #     #load binary features by default..
    #     captions = []
    #     domains = []   
    #     #if no captions, return empty 
    #     if not os.path.isfile(captions_info_path):
    #         captions = ['']
    #         domains = torch.zeros((1))
    #         captions_binary = torch.zeros((1))
    #         return captions, domains, captions_binary
    #     #load saved captions as text 
    #     captions_info = json.load(open(captions_info_path))     
    #     #load the indices of captions to keep.. if no captions to keep, return empty         
    #     captions_to_keep_idx = json.load(open(captions_to_keep_path))['index_of_captions_tokeep']
    #     if len(captions_to_keep_idx) == 0:            
    #         captions = ['']
    #         domains = torch.zeros((1))
    #         captions_binary = torch.zeros((1))
    #         return captions, domains, captions_binary  
    #     captions_to_keep_idx = np.asarray([int(i) for i in captions_to_keep_idx])
    #     all_captions = captions_info['captions']
    #     all_domains = captions_info['domains']
    #     #get captions to keep only 
    #     for i in range(0,len(captions_to_keep_idx)):
    #         captions.append(process_string(all_captions[int(captions_to_keep_idx[i])]))
    #         domain_idx = self.get_domain_idx(all_domains[str(captions_to_keep_idx[i])])
    #         domains.append(domain_idx)
    #     domains = torch.as_tensor(domains)
    #     if self.binary_feature_cap:
    #         captions_binary = np.load(captions_binary_path)['arr_0']
    #         captions_binary= captions_binary[captions_to_keep_idx]
    #         captions_binary = torch.from_numpy(captions_binary)
    #     return captions, domains, captions_binary

    def load_captions(self,inv_dict,captions_binary_path):
        captions = []
        domains = {}   
        captions = ['']
        #domains = torch.zeros((1))
        #captions_binary = torch.zeros((1))
        #return captions, domains, captions_binary
        pages_with_captions_keys = ['all_fully_matched_captions','all_partially_matched_captions']
        count = 0 
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)
                        domains[count] = urlparse(page['page_link']).netloc
                        count = count+1 
                    
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        page_domain = urlparse(page['page_link']).netloc
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 

                            domains[count] = urlparse(page['page_link']).netloc
                            count = count+1
                        captions = captions + sub_captions_list 
                    
        pages_with_title_only_keys = ['partially_matched_no_text','fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
                        domains[count] = urlparse(page['page_link']).netloc
                        count = count+1
        if self.binary_feature_cap:
            captions_binary = np.load(captions_binary_path)['arr_0']
            # if self.filter_duplicates: captions_binary=captions_binary[captions_to_keep_idx]
            captions_binary = torch.from_numpy(captions_binary) 
            #print(f'binary caption is {captions_binary},shape is {captions_binary.shape}')
        domains = torch.zeros((len(captions)))
        captions_binary = torch.zeros(1)
        return captions, domains ,captions_binary
    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()               
        key = self.idx_to_keys[idx]
        label = torch.as_tensor(int(self.context_data_items_dict[key]['label']))
        direct_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['direct_path'])
        inverse_path_item = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['inv_path'])

        #load evidence entities   
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
        entities_binary_file_path = os.path.join(inverse_path_item, 'entities_binary_feature2.npz')
        entities,entities_binary  = self.load_entities(inv_ann_dict,entities_binary_file_path)
        
        #load evidence captions 
        cap_info_file_path = os.path.join(inverse_path_item, 'captions_info')    
        cap_binary_file_path = os.path.join(inverse_path_item, 'caption_features2.npz') 
        cap_filter_mask = os.path.join(inverse_path_item, 'captions_to_keep_idx') 
        captions,captions_domains,captions_binary = self.load_captions(inv_ann_dict,cap_binary_file_path)

        #load evidence imgs   
        imgs_emb_file_path = os.path.join(direct_path_item, 'resnet_features.npz') 
        imgs_info_file_path = os.path.join(direct_path_item, 'metadata_of_features') 
        imgs_filter_mask = os.path.join(direct_path_item, 'imgs_to_keep_idx') 
        labels_overlap_file = os.path.join(inverse_path_item, 'labels_overlap_percentage.npz') 
        imgs,imgs_domains = self.load_imgs_direct_search(imgs_emb_file_path,imgs_info_file_path,imgs_filter_mask,labels_overlap_file,label)     

        #load evidence resnet_places 
        places_emb_file_path = os.path.join(direct_path_item, 'places_resnet_features.npz') 
        places_emb = self.load_imgs_direct_search(places_emb_file_path,imgs_info_file_path,imgs_filter_mask,labels_overlap_file,label,places=True)  
        #load query         
        qImg_path = os.path.join(inverse_path_item, 'qImg_resnet_features.npz')
        qImg_places_path = os.path.join(inverse_path_item, 'qImg_places_resnet_features.npz')
        qCap_path = os.path.join(inverse_path_item, 'qCap_sbert_features.npz')
        # load query caption as text 
        # visual_news_caption_item = self.visual_news_data_mapping[self.news_clip_data_dict[int(key)]["id"]]
        qCap, qImg, qPlaces =  self.load_queries(idx,qImg_path, qplace_path=qImg_places_path)
            
        # if self.load_clip_for_queries:
        #     qImg_path_clip = os.path.join(inverse_path_item, 'qImg_clip_features.npz')
        #     qCap_path_clip = os.path.join(inverse_path_item, 'qCap_clip_features.npz')
        #     qCap_clip, qImg_clip =  self.load_queries(qImg_path_clip, qcap_path=qCap_path_clip)

        sample = {'label': label, 'entities':entities, 'entities_binary':entities_binary, 'caption': captions, 'caption_domains': captions_domains, 'captions_binary': captions_binary, 'imgs': imgs.float(), 'imgs_domains': imgs_domains, 'places_mem': places_emb, 'qImg': qImg.float(), 'qCap': qCap, 'qPlaces': qPlaces}  
        # if self.load_clip_for_queries:   
        #     sample['qImg_clip'] = qImg_clip.float() 
        #     sample['qCap_clip'] = qCap_clip.float()             
        return sample, len(entities), len(captions), imgs.size(0)
