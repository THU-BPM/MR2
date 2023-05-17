CUDA_VISIBLE_DEVICES=0 python bert_lstm/main_bert_lstm.py --use_src --use_cap_memory --use_img_memory --use_ent_memory \
--use_places_memory --inp_pdrop 0.05 --pdrop_mem 0.2  --pdrop 0 --nlayers 2 --emb_pdrop 0.25 \
--lr 0.000009 --dataset_items_file ./dataset_items_  --domains_file ./domain_to_idx_dict.json  --epochs 60 \
--batch_size 32  --lstm_maxlen 150 --queries_dataset_root ../queries_dataset/merge/  