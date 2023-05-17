import torch 

def collate_all_embs(batch):
    samples = [item[0] for item in batch]
    max_entities_len = max([item[1] for item in batch])
    max_captions_len = max([item[2] for item in batch])
    max_images_len = max([item[3] for item in batch])
    entities_batch = []
    mem_cap_batch = []
    mem_cap_domains_batch = []
    mem_img_batch = []
    mem_img_domains_batch = []
    mem_places_batch = []
    qCap_batch = []
    qImg_batch = []
    qPlaces_batch = []

    qCap_clip_batch = []
    qImg_clip_batch = []
    labels = [] 
    for j in range(0,len(samples)):  
        sample = samples[j]    
        labels.append(sample['label'])
        #pad entities
        entities = sample['entities']
        padding_size = (max_entities_len-sample['entities'].size(0), sample['entities'].size(1))
        padded_mem_ent = torch.cat((sample['entities'], torch.zeros(padding_size)),dim=0)
        entities_batch.append(padded_mem_ent)
        #pad captions
        padding_size = (max_captions_len-sample['caption'].size(0), sample['caption'].size(1))
        padded_mem_cap = torch.cat((sample['caption'], torch.zeros(padding_size)),dim=0)
        mem_cap_batch.append(padded_mem_cap)
        #pad domains of captions 
        padded_cap_domains = torch.cat( (sample['caption_domains'], torch.zeros((max_captions_len-sample['caption'].size(0)))) )
        mem_cap_domains_batch.append(padded_cap_domains)
        if sample['caption'].size(0) != sample['caption_domains'].size(0):
            print('domains mismatch - captions')
        #padded images 
        if len(sample['imgs'].size()) > 2:
            padding_size = (max_images_len-sample['imgs'].size(0),sample['imgs'].size(1),sample['imgs'].size(2),sample['imgs'].size(3))
        else:
            padding_size = (max_images_len-sample['imgs'].size(0),sample['imgs'].size(1))
        if sample['imgs'].size(0) != sample['imgs_domains'].size(0):
            print('domains mismatch')
        padded_mem_img = torch.cat((sample['imgs'], torch.zeros(padding_size)),dim=0)
        mem_img_batch.append(padded_mem_img)
        #pad domains of images 
        padded_img_domains = torch.cat( (sample['imgs_domains'], torch.zeros((max_images_len-sample['imgs'].size(0)))))
        mem_img_domains_batch.append(padded_img_domains)
        #places memory
        padding_size = (max_images_len-sample['imgs'].size(0),sample['places_mem'].size(1))
        padded_mem_places = torch.cat((sample['places_mem'], torch.zeros(padding_size)),dim=0)
        mem_places_batch.append(padded_mem_places)
        #Query 
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])
        qPlaces_batch.append(sample['qPlaces'])
        
        if 'qImg_clip' in sample.keys(): qImg_clip_batch.append(sample['qImg_clip'])   
        if 'qCap_clip' in sample.keys(): qCap_clip_batch.append(sample['qCap_clip'])          
        
    #stack 
    entities_batch = torch.stack(entities_batch, dim=0)
    mem_cap_batch = torch.stack(mem_cap_batch, dim=0)
   # print(f'cap_batch is {mem_cap_batch.shape}')
    mem_cap_domains_batch = torch.stack(mem_cap_domains_batch, dim=0).long()
    mem_img_batch = torch.stack(mem_img_batch, dim=0)
   # print(f'img_batch is {mem_img_batch.shape}')
    mem_img_domains_batch = torch.stack(mem_img_domains_batch, dim=0).long()
    mem_places_batch = torch.stack(mem_places_batch, dim=0)
    qImg_batch = torch.cat(qImg_batch, dim=0) 
    qCap_batch = torch.cat(qCap_batch, dim=0)
    qPlaces_batch = torch.cat(qPlaces_batch, dim=0)

    labels = torch.stack(labels, dim=0) 
    if qImg_clip_batch and qCap_clip_batch:
        qImg_clip_batch = torch.cat(qImg_clip_batch, dim=0)
        qCap_clip_batch = torch.cat(qCap_clip_batch, dim=0)
        return labels, entities_batch, mem_cap_batch, mem_cap_domains_batch, mem_img_batch, mem_img_domains_batch, mem_places_batch, qCap_batch, qImg_batch, qPlaces_batch, qCap_clip_batch, qImg_clip_batch
    return labels, entities_batch, mem_cap_batch, mem_cap_domains_batch, mem_img_batch, mem_img_domains_batch, mem_places_batch, qCap_batch, qImg_batch, qPlaces_batch    