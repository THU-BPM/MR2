from custom_collate import collate_context_bert
import dataset_bert_lstm 
import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import io
import model_bert_lstm_inspect
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
from transformers import BertTokenizer, BertModel


parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
##### locations #####
parser.add_argument('--queries_dataset_root', type=str, default='../../queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')   
parser.add_argument('--visual_news_root', type=str, default='../../visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')
parser.add_argument('--news_clip_root', type=str, default='../../news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')               
parser.add_argument('--dataset_items_file', type=str, default='../dataset_items_',
                    help='location to the dataset items file')
parser.add_argument('--domains_file', type=str, default='../domain_to_idx_dict.json',
                    help='location to the domains to idx file')
                    
###### model details ########                    
parser.add_argument('--domains_dim', type=int, default=20,
                    help='dimension of domains embeddings')        
parser.add_argument('--mem_img_dim_out', type=int, default=1024,
                    help='dimension of image memory')
parser.add_argument('--mem_places_dim_out', type=int, default=1024,
                    help='dimension of resnet places memory')
parser.add_argument('--mem_sent_dim_out', type=int, default=512,
                    help='projection dimension of sentence emb in the memory network module')
parser.add_argument('--mem_ent_dim_out', type=int, default=512,
                    help='projection dimension of entities emb in the memory network module')
parser.add_argument('--consistency', type=str, default='san',
                    help='which method to use for mismatch between queries, options: clip, san, embeddings. otherwise, will not be used')
parser.add_argument('--pdrop', type=float, default=0.5,
                    help='dropout probability')                    
parser.add_argument('--pdrop_mem', type=float, default=0.05,
                    help='dropout probability')
parser.add_argument('--inp_pdrop', type=float, default=0.05,
                    help='dropout probability for input features')
parser.add_argument('--emb_pdrop', type=float, default=0.1,
                    help='dropout probability for the embeddings of domains')
parser.add_argument('--img_mem_hops', type=int, default=1,
                    help='number of hops for the img memory') 
parser.add_argument('--cap_mem_hops', type=int, default=1,
                    help='number of hops for the cap memory') 
parser.add_argument('--ent_mem_hops', type=int, default=1,
                    help='number of hops for the ent memory')
parser.add_argument('--places_mem_hops', type=int, default=1,
                    help='number of hops for the places memory')            
parser.add_argument('--fusion', type=str, default='byFeatures',
                    help='how to fuse the different component, options: byFeatures and byDecision')  
parser.add_argument('--nlayers', type=int, default='2',
                    help='number of fc layers of the final classifier') 
parser.add_argument("--fc_dims",nargs="*",type=int,default=[1024],
                    help='the dimensions of the fully connected classifier layers') 
parser.add_argument('--img_rep', type=str, default='pool',
                    help='how to represent images in the memory, options: pool or regions')                      
                    
parser.add_argument('--use_src', action='store_true', help='whether to use domain embeddings in the network')
parser.add_argument('--use_img_memory', action='store_true', help='whether to use img memory')
parser.add_argument('--use_ent_memory', action='store_true', help='whether to use ent memory')
parser.add_argument('--use_cap_memory', action='store_true', help='whether to use cap memory')
parser.add_argument('--use_places_memory', action='store_true', help='whether to use resnet places memory')
parser.add_argument('--binary_ner_ent', action='store_true', help='whether to compute binary feature of NE overlap between entities and query caption')
parser.add_argument('--binary_ner_cap', action='store_true', help='whether to compute binary feature of NE overlap between captions and query caption')
parser.add_argument('--labels_overlap', action='store_true', help='whether to load labels overlap between images')
parser.add_argument('--filter_dup', action='store_true', help='whether to filter out evidence that exactly match the query')

###### LSTM model details ########                    
parser.add_argument('--lstm_num_layers', type=int, default=1,
                    help='number of layers for the LSTMs') 
parser.add_argument('--lstm_hidden_size', type=int, default=256,
                    help='hidden size of the lstms') 
parser.add_argument('--lstm_maxlen', type=int, default=512,
                    help='max len of sentences lstms') 
parser.add_argument('--lstm_input_size', type=int, default=768,
                    help='dimension of input')              
parser.add_argument('--bidirectional', action='store_true', help='whether to use bidirectional lstms')
parser.add_argument('--lstm_dropout', type=float, default=0.3, help='dropout between lstm layers')
parser.add_argument('--lstm_indropout', type=float, default=0.05, help='dropout to the input of lstms')


##### Training details #####
parser.add_argument('--batch_size', type=int, default=32,
                    help='bs')
parser.add_argument('--eval_batch_size', type=int, default=32,
                    help='bs for validation and test sets')                    
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of data loaders workers')                   
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--checkpt', type=str, default = 'exp3/best_model_acc.pth.tar', help='path to model')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def bert_tokenizer_get_maxlen(array_of_arrays):
    lengths_arr = []
    for evidence_set in array_of_arrays:
        for one_caption in evidence_set:
            tokenize_out = tokenizer.tokenize(one_caption)
            lengths_arr.append(len(tokenize_out))
    if len(lengths_arr)==0: return 2 
    max_len = max(lengths_arr)+2  #2 special tokens 
    if max_len > args.lstm_maxlen: max_len = args.lstm_maxlen
    return max_len
    
def bert_tokenizer(text, max_len):
    #text is an array of sentences 
    input_ids = []
    attention_masks = []
    type_ids = []
    for sent in text:
        tokenize_out = tokenizer.encode_plus(sent,add_special_tokens = True, truncation = True, padding = 'max_length', max_length = max_len, return_tensors = 'pt')
        input_ids.append(tokenize_out['input_ids'])
        attention_masks.append(tokenize_out['attention_mask'])
        type_ids.append(tokenize_out['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0).cuda()
    type_ids = torch.cat(type_ids, dim=0).cuda()
    attention_masks = torch.cat(attention_masks, dim=0).cuda()
    return input_ids, type_ids, attention_masks 

def bert_embeddings(input_ids, type_ids, attention_masks, embeddings_type = 'second_to_last'):
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, token_type_ids=type_ids, attention_mask=attention_masks, output_hidden_states=True)
    hidden_states = outputs[2]
    if embeddings_type == 'second_to_last':
        token_embeddings_second_to_last = hidden_states[-2]
        return token_embeddings_second_to_last        
    token_embeddings = torch.stack(hidden_states, dim=0)
    if embeddings_type == 'stacked': 
        return token_embeddings
    if embeddings_type == 'avg_four': 
        token_embeddings_last4_avg = torch.mean(token_embeddings[13-4:13], dim=0)
        return token_embeddings_last4_avg 

def evidence_BertEmbs_batch(array_of_arrays,embeddings_type='second_to_last'):
    #input is an array of arrays. each sub-array contains the evidence for that example.    
    max_tokens_len = bert_tokenizer_get_maxlen(array_of_arrays)
    evidence_embs_batch = []
    for evidence_set in array_of_arrays:
        input_ids, type_ids, attention_masks = bert_tokenizer(evidence_set, max_tokens_len)
        one_evidence_embs = bert_embeddings(input_ids, type_ids, attention_masks, embeddings_type) 
        evidence_embs_batch.append(one_evidence_embs)
    if len(evidence_embs_batch) == 0:
        print('Empty')
        print(array_of_arrays)
    evidence_embs_batch = torch.stack(evidence_embs_batch, dim=0)
    return evidence_embs_batch 

def query_BertEmbs_batch(array_of_text,embeddings_type='second_to_last'):
    #input is array of strings with the size of the batch
    max_tokens_len = bert_tokenizer_get_maxlen([array_of_text])
    input_ids, type_ids, attention_masks = bert_tokenizer(array_of_text, max_tokens_len)
    query_embs = bert_embeddings(input_ids, type_ids, attention_masks, embeddings_type) 
    return query_embs 
   

#### load input files ####
data_items_train = json.load(open(args.dataset_items_file+"train.json"))
data_items_val = json.load(open(args.dataset_items_file+"val.json"))
data_items_test = json.load(open(args.dataset_items_file+"test.json"))
domain_to_idx_dict = json.load(open(args.domains_file))

#### load Datasets and DataLoader ####
sent_emb_dim = args.lstm_hidden_size*2 * (2 if args.bidirectional else 1) 

val_dataset = dataset_bert_lstm.NewsContextDatasetEmbs(data_items_val, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'val', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' else False, \
labels_overlap=args.labels_overlap,\
binary_feature_ent=args.binary_ner_ent, binary_feature_cap=args.binary_ner_cap,filter_duplicates=args.filter_dup)

test_dataset = dataset_bert_lstm.NewsContextDatasetEmbs(data_items_test, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'test', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' else False, \
labels_overlap=args.labels_overlap,\
binary_feature_ent=args.binary_ner_ent, binary_feature_cap=args.binary_ner_cap,filter_duplicates=args.filter_dup)

val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn = collate_context_bert, num_workers=args.num_workers,  pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn = collate_context_bert, num_workers=args.num_workers,  pin_memory=True)

#### settings of the model ####
img_features_dim = 2048 #resnet dimension 
places_mem_dim = 2048 #resnet places dim 

model_settings = {'use_img_memory':args.use_img_memory, 'use_cap_memory':args.use_cap_memory, 'use_ent_memory': args.use_ent_memory,\
'use_src': args.use_src, 'use_places_memory':args.use_places_memory,'domains_num': len(domain_to_idx_dict), 'domains_dim': args.domains_dim, \
'img_dim_in': img_features_dim, 'img_dim_out': args.mem_img_dim_out, 'places_dim_in': places_mem_dim, 'places_dim_out': args.mem_places_dim_out, \
'ent_dim_in': sent_emb_dim,'ent_dim_out': args.mem_ent_dim_out, 'sent_emb_dim_in': sent_emb_dim,'sent_emb_dim_out': args.mem_sent_dim_out, \
'consistency': args.consistency, \
'fusion': args.fusion, 'pdrop': args.pdrop, 'inp_pdrop': args.inp_pdrop, 'pdrop_mem': args.pdrop_mem, 'emb_pdrop':args.emb_pdrop, \
'nlayers': args.nlayers, 'fc_dims': args.fc_dims, \
'img_mem_hops': args.img_mem_hops, 'cap_mem_hops': args.cap_mem_hops, 'ent_mem_hops': args.ent_mem_hops, 'places_mem_hops':args.places_mem_hops,\
'binary_feature_cap':args.binary_ner_cap, 'binary_feature_ent':args.binary_ner_ent,'labels_overlap':args.labels_overlap}

lstm_settings = {'num_layers': args.lstm_num_layers, 'bidirectional':args.bidirectional,\
'hidden_size': args.lstm_hidden_size, 'input_size': args.lstm_input_size,\
'dropout': args.lstm_dropout, 'dropout_in': args.lstm_indropout}

model = model_bert_lstm_inspect.ContextMemNet(model_settings,lstm_settings)
model.cuda()

#define loss function
criterion = nn.BCEWithLogitsLoss().cuda()

#the saved features are not average-pooled
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.cuda()
for param in bert_model.parameters():
    param.requires_grad = False

def evaluate(loader,split):
    total_loss = 0
    correct = 0
    correct_falsified = 0
    correct_match = 0  
    
    total_num = 0
    total_num_falsified = 0
    total_num_match = 0
    
    model.eval()
    bert_model.eval()
    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda()
        entities_text = batch[1]
        captions_text = batch[2] 
        captions_domains = batch[3].cuda() if (args.use_src and args.use_cap_memory) else None 
        img_mem = batch[4].cuda() if args.use_img_memory else None 
        img_mem_domains = batch[5].cuda() if ( (args.use_img_memory or args.use_places_memory) and args.use_src) else None 
        places_mem = batch[6].cuda() if args.use_places_memory else None 
        qCap_text = batch[7]
        qImg = batch[8].cuda()
        qPlace = batch[9].cuda() if args.use_places_memory else None 
        entities_binary = batch[10].cuda() if args.binary_ner_ent else None 
        captions_binary = batch[11].cuda() if args.binary_ner_cap else None 
        if args.consistency == 'clip':     
            qCap_clip = batch[12].cuda()  
            qImg_clip = batch[13].cuda()   
        else:       
            qCap_clip = None      
            qImg_clip = None
        with torch.no_grad():
            entities_embs = evidence_BertEmbs_batch(entities_text) if args.use_ent_memory else None 
            captions_embs = evidence_BertEmbs_batch(captions_text) if args.use_cap_memory else None 
            qCap_embs = query_BertEmbs_batch(qCap_text)
            img_dim = (model_settings['img_dim_in']+1) if args.labels_overlap else model_settings['img_dim_in']
            qImg_avg = adaptive_pool(qImg).view(labels.size(0),model_settings['img_dim_in']) 
            if img_mem is not None: 
                num_img_mem = img_mem.size(1) 
                if args.img_rep == 'pool':
                    img_mem = adaptive_pool(img_mem).view(labels.size(0),num_img_mem,img_dim)
                elif args.img_rep == 'regions': 
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7*7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7*7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem*7*7, img_dim)
                    if img_mem_domains is not None: img_mem_domains = img_mem_domains.unsqueeze(dim=2).repeat(1,1,49).view(labels.size(0),num_img_mem*49)


            #forward 
            output = model(qImg_avg, qCap_embs, qtext_clip = qCap_clip, qimage_clip=qImg_clip, query_places=qPlace,\
                       entities_bert=entities_embs, entities_binary=entities_binary,\
                       results_images=img_mem, results_places = places_mem, images_domains=img_mem_domains, \
                       results_captions_bert=captions_embs, captions_domains=captions_domains, captions_binary=captions_binary) 
            #save for inspection
            np.savez('attn_weights/batch_'+str(idx)+'_'+split+'.npz', name1=torch.sigmoid(output[0]).cpu().detach().numpy(),\
            name2=output[1].cpu().detach().numpy(),name3=output[2].cpu().detach().numpy(),name4=output[3].cpu().detach().numpy(),\
            name5=output[4].cpu().detach().numpy())   
            
            #compute loss     
            loss = criterion(output[0], torch.unsqueeze(labels.float(), 1))
            total_loss += loss.item()
            #compute correct predictions   
            pred = torch.sigmoid(output[0]) >= 0.5
            truth = torch.unsqueeze(labels,1)>=0.5
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0) 

            #print(labels)            
            index_ones = ((labels == 1).nonzero(as_tuple=True)[0])
            correct_falsified += pred[index_ones,:].eq(truth[index_ones,:]).sum().item()  
            total_num_falsified += index_ones.size(0)

            index_zeros = ((labels == 0).nonzero(as_tuple=True)[0])
            correct_match += pred[index_zeros,:].eq(truth[index_zeros,:]).sum().item()   
            total_num_match += index_zeros.size(0)
            
    avg_loss = total_loss/len(loader)    
    acc = (correct/total_num)*100
    acc_falsified = (correct_falsified/total_num_falsified)*100
    acc_match = (correct_match/total_num_match)*100
    return avg_loss, acc, acc_falsified, acc_match

print("=> loading checkpoint '{}'".format(args.checkpt))
checkpoint = torch.load(args.checkpt)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpt, checkpoint['epoch']))

val_loss, val_acc, val_acc_false, val_acc_match = evaluate(val_dataloader,'val')
print('=' * 89)
print('| val loss {:5.2f} | val acc {:8.3f} | val falsified acc {:8.3f} | val match acc {:8.3f}'.format(
val_loss, val_acc,val_acc_false,val_acc_match))
print('=' * 89)

test_loss, test_acc, test_acc_false, test_acc_match = evaluate(test_dataloader,'test')
print('=' * 89)
print('| test loss {:5.2f} | test acc {:8.3f} | test falsified acc {:8.3f} | test match acc {:8.3f}'.format(
test_loss, test_acc,test_acc_false,test_acc_match))
print('=' * 89)
    

