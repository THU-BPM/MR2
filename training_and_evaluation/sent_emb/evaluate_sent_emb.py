from custom_collate import collate_context_places
import dataset_sent_emb 
import json 
import torch
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import model_sent_emb_inspect
import torch.nn.parallel
import torch.optim

parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
##### locations #####
parser.add_argument('--queries_dataset_root', type=str, default='../queries_dataset/merged_balanced/',
                    help='location to the root folder of the query dataset')   
parser.add_argument('--visual_news_root', type=str, default='../visual_news/origin/',
                    help='location to the root folder of the visualnews dataset')
parser.add_argument('--news_clip_root', type=str, default='../news_clippings/data/merged_balanced/',
                    help='location to the root folder of the clip dataset')               
parser.add_argument('--dataset_items_file', type=str, default='dataset_items_',
                    help='location to the dataset items file')
parser.add_argument('--domains_file', type=str, default='domain_to_idx_dict.json',
                    help='location to the domains to idx file')
                    
###### model details ########                    
parser.add_argument('--domains_dim', type=int, default=20,
                    help='dimension of domains embeddings')        
parser.add_argument('--mem_img_dim_out', type=int, default=1024,
                    help='dimension of image memory')
parser.add_argument('--mem_places_dim_out', type=int, default=1024,
                    help='dimension of resnet places memory')
parser.add_argument('--mem_sent_dim_out', type=int, default=768,
                    help='projection dimension of sentence emb in the memory network module')
parser.add_argument('--mem_ent_dim_out', type=int, default=768,
                    help='projection dimension of entities emb in the memory network module')
parser.add_argument('--consistency', type=str, default='san',
                    help='which method to use for mismatch between queries, options: clip, san, embeddings. otherwise, will not be used')
parser.add_argument('--san_emb', type=int, default=1024,
                    help='projection dimension of san')
parser.add_argument('--consis_proj_dim', type=int, default=1024,
                    help='projection dimension of using the embeddings directly for consistency prediction')
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

parser.add_argument('--shared_cap_proj', action='store_true', help='whether to use the same project for the caption in the caption and entities memories')
parser.add_argument('--shared_cap_ent_mem', action='store_true', help='whether to use the share the caption and entities memories')
parser.add_argument('--use_clip_for_all', action='store_true', help='whether to use clip for all embeddings')
parser.add_argument('--binary_ner_ent', action='store_true', help='whether to compute binary feature of NE overlap between entities and query caption')
parser.add_argument('--binary_ner_cap', action='store_true', help='whether to compute binary feature of NE overlap between captions and query caption')
parser.add_argument('--labels_overlap', action='store_true', help='whether to load labels overlap between images')
parser.add_argument('--filter_dup', action='store_true', help='whether to filter out evidence that exactly match the query')


##### Training details #####
parser.add_argument('--batch_size', type=int, default=64,
                    help='dimension of domains embeddings') 
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of data loaders workers') 
parser.add_argument('--checkpt', type=str, default = 'exp/best_model_acc.pth.tar', help='path to model')                      
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
                    
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### load input files ####
data_items_val = json.load(open(args.dataset_items_file+"val.json"))
data_items_test = json.load(open(args.dataset_items_file+"test.json"))
domain_to_idx_dict = json.load(open(args.domains_file))

#### load Datasets and DataLoader ####
sent_emb_dim = 512 if args.use_clip_for_all else 768 #sentence bert dimension 

val_dataset = dataset_sent_emb.NewsContextDatasetEmbs(data_items_val, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'val', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' and not args.use_clip_for_all else False, \
load_clip_for_all = args.use_clip_for_all,labels_overlap=args.labels_overlap,\
binary_feature_ent=args.binary_ner_ent, binary_feature_cap=args.binary_ner_cap,filter_duplicates=args.filter_dup)

test_dataset = dataset_sent_emb.NewsContextDatasetEmbs(data_items_test, args.visual_news_root, args.queries_dataset_root, args.news_clip_root,\
domain_to_idx_dict, 'test', sent_emb_dim, load_clip_for_queries=True if args.consistency=='clip' and not args.use_clip_for_all else False, \
load_clip_for_all = args.use_clip_for_all,labels_overlap=args.labels_overlap,\
binary_feature_ent=args.binary_ner_ent, binary_feature_cap=args.binary_ner_cap,filter_duplicates=args.filter_dup)

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_context_places, num_workers=args.num_workers,  pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_context_places, num_workers=args.num_workers,  pin_memory=True)

#### settings of the model ####
img_features_dim = 512 if args.use_clip_for_all else 2048 #resnet dimension 

places_mem_dim = 2048 #resnet places dim 

model_settings = {'use_img_memory':args.use_img_memory, 'use_cap_memory':args.use_cap_memory, 'use_ent_memory': args.use_ent_memory,\
'use_src': args.use_src, 'use_places_memory':args.use_places_memory,'domains_num': len(domain_to_idx_dict), 'domains_dim': args.domains_dim, \
'img_dim_in': img_features_dim, 'img_dim_out': args.mem_img_dim_out, 'places_dim_in': places_mem_dim, 'places_dim_out': args.mem_places_dim_out, \
'ent_dim_in': sent_emb_dim,'ent_dim_out': args.mem_ent_dim_out, 'sent_emb_dim_in': sent_emb_dim,'sent_emb_dim_out': args.mem_sent_dim_out, \
'consistency': args.consistency, 'san_emb': args.san_emb, 'consis_proj_dim': args.consis_proj_dim, \
'fusion': args.fusion, 'pdrop': args.pdrop, 'inp_pdrop': args.inp_pdrop, 'pdrop_mem': args.pdrop_mem, 'emb_pdrop':args.emb_pdrop, \
'nlayers': args.nlayers, 'fc_dims': args.fc_dims, \
'img_mem_hops': args.img_mem_hops, 'cap_mem_hops': args.cap_mem_hops, 'ent_mem_hops': args.ent_mem_hops, 'places_mem_hops':args.places_mem_hops,\
'use_clip_for_all': args.use_clip_for_all,\
'binary_feature_cap':args.binary_ner_cap, 'binary_feature_ent':args.binary_ner_ent,'labels_overlap':args.labels_overlap}

model = model_sent_emb_inspect.ContextMemNet(model_settings)
model.cuda()

#define loss function
criterion = nn.BCEWithLogitsLoss().cuda()

#the saved features are not average-pooled
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

def evaluate(loader,split):
    total_loss = 0
    correct = 0
    correct_falsified = 0
    correct_match = 0     
    total_num = 0
    total_num_falsified = 0
    total_num_match = 0
    model.eval()
    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda()
        print(labels)
        entities_mem = batch[1].cuda() if args.use_ent_memory else None 
        captions = batch[2].cuda() if args.use_cap_memory else None 
        captions_domains = batch[3].cuda() if (args.use_src and args.use_cap_memory) else None 
        img_mem = batch[4].cuda() if args.use_img_memory else None 
        img_mem_domains = batch[5].cuda() if (args.use_img_memory and args.use_src) else None 
        places_mem = batch[6].cuda() if args.use_places_memory else None 
        qCap = batch[7].cuda()
        qImg = batch[8].cuda()
        qPlace = batch[9].cuda() if args.use_places_memory else None 
        
        if args.consistency == 'clip' and not args.use_clip_for_all:     
            qCap_clip = batch[10].cuda()  
            qImg_clip = batch[11].cuda()   
        else:       
            qCap_clip = None      
            qImg_clip = None
        with torch.no_grad():
            img_dim = (model_settings['img_dim_in']+1) if args.labels_overlap else model_settings['img_dim_in']
            if not args.use_clip_for_all: qImg_avg = adaptive_pool(qImg).view(labels.size(0),model_settings['img_dim_in']) 
            if img_mem is not None: 
                num_img_mem = img_mem.size(1) 
                if args.img_rep == 'pool' and not args.use_clip_for_all:
                    img_mem = adaptive_pool(img_mem).view(labels.size(0),num_img_mem,img_dim)
                elif args.img_rep == 'regions': 
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7*7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7*7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem*7*7, img_dim)
                    if img_mem_domains is not None: img_mem_domains = img_mem_domains.unsqueeze(dim=2).repeat(1,1,49).view(labels.size(0),num_img_mem*49)


            #forward 
            output = model(qImg_avg if not args.use_clip_for_all else qImg, qCap, query_places=qPlace, entities=entities_mem, \
                       results_images=img_mem, results_places = places_mem, images_domains=img_mem_domains, \
                       results_captions=captions, captions_domains=captions_domains, \
                       query_img_regions = qImg if args.consistency == 'san' else None, qtext_clip = qCap_clip, qimage_clip=qImg_clip) 
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
    

