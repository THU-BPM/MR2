import torch 


def collate_context_bert(batch):
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
    captions_binary_batch = []
    entities_binary_batch = []

    qCap_clip_batch = []
    qImg_clip_batch = []
    labels = [] 
    for j in range(0,len(samples)):  
        sample = samples[j]    
        labels.append(sample['label'])
        #pad entities
        entities = sample['entities']
        ent_len = len(entities)
        for i in range(0,max_entities_len-ent_len):
            entities.append("")
        entities_batch.append(entities)
        #pad entities binary 
        entities_binary = sample['entities_binary']
        entities_binary = torch.cat( (entities_binary.unsqueeze(1), torch.zeros((max_entities_len-ent_len,1))),dim=0)
        entities_binary_batch.append(entities_binary)
        #pad captions
        if len(sample['caption']) != sample['caption_domains'].size(0):
            print('domains mismatch - captions')
            print(len(sample['caption']))
            print(sample['caption_domains'].size(0))  
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        mem_cap_batch.append(captions)
        #pad captions binary
        captions_binary = sample['captions_binary']
        captions_binary = torch.cat( (captions_binary.unsqueeze(1), torch.zeros((max_captions_len-cap_len,1))),dim=0)
        captions_binary_batch.append(captions_binary)
        #pad domains of captions 
        padded_cap_domains = torch.cat( (sample['caption_domains'], torch.zeros((max_captions_len-cap_len))) )
        mem_cap_domains_batch.append(padded_cap_domains)
        #padded images 
        if sample['imgs'].size(0) != sample['imgs_domains'].size(0):
            print('domains mismatch')
        if len(sample['imgs'].size()) > 2:
            padding_size = (max_images_len-sample['imgs'].size(0),sample['imgs'].size(1),sample['imgs'].size(2),sample['imgs'].size(3))
        else:
            padding_size = (max_images_len-sample['imgs'].size(0),sample['imgs'].size(1))

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
    mem_cap_domains_batch = torch.stack(mem_cap_domains_batch, dim=0).long()
    mem_img_batch = torch.stack(mem_img_batch, dim=0)
    mem_img_domains_batch = torch.stack(mem_img_domains_batch, dim=0).long()
    mem_places_batch = torch.stack(mem_places_batch, dim=0)
    #captions_binary_batch = torch.stack(captions_binary_batch, dim=0)
    entities_binary_batch = torch.stack(entities_binary_batch, dim=0)
    
    qImg_batch = torch.cat(qImg_batch, dim=0)
    qPlaces_batch = torch.cat(qPlaces_batch, dim=0)

    labels = torch.stack(labels, dim=0) 
    if qImg_clip_batch and qCap_clip_batch:
        qImg_clip_batch = torch.cat(qImg_clip_batch, dim=0)
        qCap_clip_batch = torch.cat(qCap_clip_batch, dim=0)
        return labels, entities_batch, mem_cap_batch, mem_cap_domains_batch, mem_img_batch, mem_img_domains_batch, mem_places_batch, qCap_batch, qImg_batch, qPlaces_batch, entities_binary_batch, captions_binary_batch, qCap_clip_batch, qImg_clip_batch
    #print(f'qimg shape is {qImg_batch.shape}')
    return labels, entities_batch, mem_cap_batch, mem_cap_domains_batch, mem_img_batch, mem_img_domains_batch, mem_places_batch, qCap_batch, qImg_batch, qPlaces_batch, entities_binary_batch, None

   