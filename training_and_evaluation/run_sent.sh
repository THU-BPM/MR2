CUDA_VISIBLE_DEVICES=0 python -u sent_emb/main_sent_emb.py --use_src --use_cap_memory --use_img_memory \
--use_ent_memory --use_places_memory --inp_pdrop 0.05 --pdrop_mem 0.1 \
--pdrop 0 --nlayers 2 --domains_dim 20 --emb_pdrop 0.25 --lr 0.000009 \
--epochs 60    \
--queries_dataset_root ../queries_dataset/merge/ \
--dataset_items_file ./dataset_items_  --domains_file ./domain_to_idx_dict.json 