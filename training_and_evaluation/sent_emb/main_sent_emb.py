from custom_collate import collate_all_embs
import dataset_sent_emb
import json
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import argparse
import model_sent_emb
import torch.nn.parallel
import torch.optim
import numpy as np
import time
from sklearn.metrics import f1_score,classification_report

parser = argparse.ArgumentParser(
    description="Training using the precomputed embeddings"
)
##### locations #####
parser.add_argument(
    "--queries_dataset_root",
    type=str,
    default="../../queries_dataset/merged_balanced/",
    help="location to the root folder of the query dataset",
)
parser.add_argument(
    "--visual_news_root",
    type=str,
    default="../visual_news/origin/",
    help="location to the root folder of the visualnews dataset",
)
parser.add_argument(
    "--news_clip_root",
    type=str,
    default="../news_clippings/data/merged_balanced/",
    help="location to the root folder of the clip dataset",
)
parser.add_argument(
    "--dataset_items_file",
    type=str,
    default="dataset_items_",
    help="location to the dataset items file",
)
parser.add_argument(
    "--domains_file",
    type=str,
    default="domain_to_idx_dict.json",
    help="location to the domains to idx file",
)
parser.add_argument(
    "--exp_folder",
    type=str,
    default="./exp/",
    help="path to the folder to log the output and save the models",
)

###### model details ########
parser.add_argument(
    "--domains_dim", type=int, default=20, help="dimension of domains embeddings"
)
parser.add_argument(
    "--mem_img_dim_out", type=int, default=1024, help="dimension of image memory"
)
parser.add_argument(
    "--mem_places_dim_out",
    type=int,
    default=1024,
    help="dimension of resnet places memory",
)
parser.add_argument(
    "--mem_sent_dim_out",
    type=int,
    default=768,
    help="projection dimension of sentence emb in the memory network module",
)
parser.add_argument(
    "--mem_ent_dim_out",
    type=int,
    default=768,
    help="projection dimension of entities emb in the memory network module",
)
parser.add_argument(
    "--consistency",
    type=str,
    default="san",
    help="which method to use for mismatch between queries, options: clip, san, embeddings. otherwise, will not be used",
)
parser.add_argument(
    "--san_emb", type=int, default=1024, help="projection dimension of san"
)
parser.add_argument(
    "--consis_proj_dim",
    type=int,
    default=1024,
    help="projection dimension of using the embeddings directly for consistency prediction",
)
parser.add_argument("--pdrop", type=float, default=0.5, help="dropout probability")
parser.add_argument("--pdrop_mem", type=float, default=0.05, help="dropout probability")
parser.add_argument(
    "--inp_pdrop",
    type=float,
    default=0.05,
    help="dropout probability for input features",
)
parser.add_argument(
    "--emb_pdrop",
    type=float,
    default=0.1,
    help="dropout probability for the embeddings of domains",
)
parser.add_argument(
    "--img_mem_hops", type=int, default=1, help="number of hops for the img memory"
)
parser.add_argument(
    "--cap_mem_hops", type=int, default=1, help="number of hops for the cap memory"
)
parser.add_argument(
    "--ent_mem_hops", type=int, default=1, help="number of hops for the ent memory"
)
parser.add_argument(
    "--places_mem_hops",
    type=int,
    default=1,
    help="number of hops for the places memory",
)


parser.add_argument(
    "--fusion",
    type=str,
    default="byFeatures",
    help="how to fuse the different component, options: byFeatures and byDesion",
)
parser.add_argument(
    "--nlayers",
    type=int,
    default="2",
    help="number of fc layers of the final classifier",
)
parser.add_argument(
    "--fc_dims",
    nargs="*",
    type=int,
    default=[1024],
    help="the dimensions of the fully connected classifier layers",
)
parser.add_argument(
    "--img_rep",
    type=str,
    default="pool",
    help="how to represent images in the memory, options: pool or regions",
)

parser.add_argument(
    "--binary_classify", action="store_true", help="whether to use binary classify"
)
parser.add_argument(
    "--use_src",
    action="store_true",
    help="whether to use domain embeddings in the network",
)
parser.add_argument(
    "--use_img_memory", action="store_true", help="whether to use img memory"
)
parser.add_argument(
    "--use_ent_memory", action="store_true", help="whether to use ent memory"
)
parser.add_argument(
    "--use_cap_memory", action="store_true", help="whether to use cap memory"
)
parser.add_argument(
    "--use_places_memory",
    action="store_true",
    help="whether to use resnet places memory",
)
parser.add_argument("--relation_learning", action="store_true")
parser.add_argument(
    "--shared_cap_proj",
    action="store_true",
    help="whether to use the same project for the caption in the caption and entities memories",
)
parser.add_argument(
    "--shared_cap_ent_mem",
    action="store_true",
    help="whether to use the share the caption and entities memories",
)
parser.add_argument(
    "--use_clip_for_all",
    action="store_true",
    help="whether to use clip for all embeddings",
)
parser.add_argument(
    "--binary_ner_ent",
    action="store_true",
    help="whether to compute binary feature of NE overlap between entities and query caption",
)
parser.add_argument(
    "--binary_ner_cap",
    action="store_true",
    help="whether to compute binary feature of NE overlap between captions and query caption",
)
parser.add_argument(
    "--labels_overlap",
    action="store_true",
    help="whether to load labels overlap between images",
)
parser.add_argument(
    "--filter_dup",
    action="store_true",
    help="whether to filter out evidence that exactly match the query",
)


##### Training details #####
parser.add_argument("--batch_size", type=int, default=64, help="bs")
parser.add_argument(
    "--eval_batch_size", type=int, default=64, help="bs for validation and test sets"
)
parser.add_argument(
    "--num_workers", type=int, default=6, help="number of data loaders workers"
)
parser.add_argument("--epochs", type=int, default=300, help="number of epochs to run")
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts) (default: 0)",
)
parser.add_argument("--log_interval", type=int, default=200, help="how many batches")
parser.add_argument("--resume", type=str, default="", help="path to model")
parser.add_argument(
    "--optimizer", type=str, default="adam", help="which optimizer to use"
)
parser.add_argument(
    "--lr", type=float, default=0.00008, help="start or base learning rate"
)
parser.add_argument("--lr_max", type=float, default=0.00008, help="learning rate")
parser.add_argument("--lr_sched", type=str, default="", help="use cyclic scheduler")
parser.add_argument("--sgd_momentum", type=float, default=0.9, help="learning rate")
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument(
    "--wdecay", default=1.2e-6, type=float, help="weight decay pow (default: -5)"
)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### load input files ####
data_items_train = json.load(open(args.dataset_items_file + "train.json"))
data_items_val = json.load(open(args.dataset_items_file + "val.json"))
data_items_test = json.load(open(args.dataset_items_file + "test.json"))
domain_to_idx_dict = json.load(open(args.domains_file))

#### load Datasets and DataLoader ####
sent_emb_dim = 512 if args.use_clip_for_all else 768  # sentence bert dimension

train_dataset = dataset_sent_emb.NewsContextDatasetEmbs(
    data_items_train,
    args.visual_news_root,
    args.queries_dataset_root,
    args.news_clip_root,
    domain_to_idx_dict,
    "train",
    sent_emb_dim,
    load_clip_for_queries=True
    if args.consistency == "clip" and not args.use_clip_for_all
    else False,
    load_clip_for_all=args.use_clip_for_all,
    labels_overlap=args.labels_overlap,
    binary_feature_ent=args.binary_ner_ent,
    binary_feature_cap=args.binary_ner_cap,
    filter_duplicates=args.filter_dup,
)

val_dataset = dataset_sent_emb.NewsContextDatasetEmbs(
    data_items_val,
    args.visual_news_root,
    args.queries_dataset_root,
    args.news_clip_root,
    domain_to_idx_dict,
    "val",
    sent_emb_dim,
    load_clip_for_queries=True
    if args.consistency == "clip" and not args.use_clip_for_all
    else False,
    load_clip_for_all=args.use_clip_for_all,
    labels_overlap=args.labels_overlap,
    binary_feature_ent=args.binary_ner_ent,
    binary_feature_cap=args.binary_ner_cap,
    filter_duplicates=args.filter_dup,
)

test_dataset = dataset_sent_emb.NewsContextDatasetEmbs(
    data_items_test,
    args.visual_news_root,
    args.queries_dataset_root,
    args.news_clip_root,
    domain_to_idx_dict,
    "test",
    sent_emb_dim,
    load_clip_for_queries=True
    if args.consistency == "clip" and not args.use_clip_for_all
    else False,
    load_clip_for_all=args.use_clip_for_all,
    labels_overlap=args.labels_overlap,
    binary_feature_ent=args.binary_ner_ent,
    binary_feature_cap=args.binary_ner_cap,
    filter_duplicates=args.filter_dup,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_all_embs,
    num_workers=args.num_workers,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.eval_batch_size,
    shuffle=False,
    collate_fn=collate_all_embs,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.eval_batch_size,
    shuffle=False,
    collate_fn=collate_all_embs,
    num_workers=args.num_workers,
    pin_memory=True,
)

#### settings of the model ####
img_features_dim = 512 if args.use_clip_for_all else 2048  # resnet dimension
places_mem_dim = 2048  # resnet places dim

# 'domains_num': len(domain_to_idx_dict)
model_settings = {
    "binary_classify": args.binary_classify,
    "use_img_memory": args.use_img_memory,
    "use_cap_memory": args.use_cap_memory,
    "use_ent_memory": args.use_ent_memory,
    "use_src": args.use_src,
    "use_places_memory": args.use_places_memory,
    "domains_num": 1,
    "domains_dim": args.domains_dim,
    "img_dim_in": img_features_dim,
    "img_dim_out": args.mem_img_dim_out,
    "places_dim_in": places_mem_dim,
    "places_dim_out": args.mem_places_dim_out,
    "ent_dim_in": sent_emb_dim,
    "ent_dim_out": args.mem_ent_dim_out,
    "sent_emb_dim_in": sent_emb_dim,
    "sent_emb_dim_out": args.mem_sent_dim_out,
    "consistency": args.consistency,
    "san_emb": args.san_emb,
    "consis_proj_dim": args.consis_proj_dim,
    "fusion": args.fusion,
    "pdrop": args.pdrop,
    "inp_pdrop": args.inp_pdrop,
    "pdrop_mem": args.pdrop_mem,
    "emb_pdrop": args.emb_pdrop,
    "nlayers": args.nlayers,
    "fc_dims": args.fc_dims,
    "img_mem_hops": args.img_mem_hops,
    "cap_mem_hops": args.cap_mem_hops,
    "ent_mem_hops": args.ent_mem_hops,
    "places_mem_hops": args.places_mem_hops,
    "use_clip_for_all": args.use_clip_for_all,
    "binary_feature_cap": args.binary_ner_cap,
    "binary_feature_ent": args.binary_ner_ent,
    "labels_overlap": args.labels_overlap,
    "relation_learning": args.relation_learning,
}


model = model_sent_emb.ContextMemNet(model_settings)
model.cuda()
# create exp folder if it has not been created
if not os.path.isdir(args.exp_folder):
    os.makedirs(args.exp_folder)

# define optimizer
params = list(model.parameters())
total_params = sum(
    x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0]
    for x in params
    if x.size()
)
print("Args:", args)
print("Model total parameters:", total_params)

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        params, lr=args.lr, weight_decay=args.wdecay, momentum=args.sgd_momentum
    )
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

# resume training
stored_loss = 100000000
stored_acc = 0
if args.resume:
    if os.path.isfile(args.resume):
        log_file_val_loss = open(os.path.join(args.exp_folder, "log_file_val.txt"), "a")
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        if "best_val_loss" in checkpoint:
            stored_loss = checkpoint["best_val_loss"]
        if "best_val_acc" in checkpoint:
            stored_acc = checkpoint["best_val_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    log_file_val_loss = open(os.path.join(args.exp_folder, "log_file_val.txt"), "w")

if args.lr_sched == "cycle":
    lr_tot_steps = args.epochs * len(train_dataloader)
    steps_up = lr_tot_steps / 3
    steps_down = 2 * lr_tot_steps / 3
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        args.lr,
        args.lr_max,
        step_size_up=steps_up,
        step_size_down=steps_down,
        cycle_momentum=False,
    )
else:
    scheduler = None
# define loss function
# change
criterion = (
    nn.MSELoss().cuda() if args.binary_classify else nn.CrossEntropyLoss().cuda()
)
print(criterion)
# the saved features are not average-pooled
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))


def evaluate(loader):
    total_loss = 0
    correct = 0
    total_num = 0
    total_pred = []
    total_truth = []
    model.eval()
    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda()
        entities_mem = batch[1].cuda() if args.use_ent_memory else None
        captions = batch[2].cuda() if args.use_cap_memory else None
        captions_domains = (
            batch[3].cuda() if (args.use_src and args.use_cap_memory) else None
        )
        img_mem = batch[4].cuda() if args.use_img_memory else None
        img_mem_domains = (
            batch[5].cuda()
            if ((args.use_img_memory or args.use_places_memory) and args.use_src)
            else None
        )
        places_mem = batch[6].cuda() if args.use_places_memory else None
        qCap = batch[7].cuda()
        qImg = batch[8].cuda()
        qPlace = batch[9].cuda() if args.use_places_memory else None

        if args.consistency == "clip" and not args.use_clip_for_all:
            qCap_clip = batch[10].cuda()
            qImg_clip = batch[11].cuda()
        else:
            qCap_clip = None
            qImg_clip = None
        with torch.no_grad():
            img_dim = (
                (model_settings["img_dim_in"] + 1)
                if args.labels_overlap
                else model_settings["img_dim_in"]
            )
            if not args.use_clip_for_all:
                qImg_avg = adaptive_pool(qImg).view(
                    labels.size(0), model_settings["img_dim_in"]
                )
            if img_mem is not None:
                num_img_mem = img_mem.size(1)
                if args.img_rep == "pool" and not args.use_clip_for_all:
                    img_mem = adaptive_pool(img_mem).view(
                        labels.size(0), num_img_mem, img_dim
                    )
                elif args.img_rep == "regions":
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7 * 7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7 * 7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem * 7 * 7, img_dim)
                    if img_mem_domains is not None:
                        img_mem_domains = (
                            img_mem_domains.unsqueeze(dim=2)
                            .repeat(1, 1, 49)
                            .view(labels.size(0), num_img_mem * 49)
                        )

            # forward
            output = model(
                qImg_avg if not args.use_clip_for_all else qImg,
                qCap,
                query_places=qPlace,
                entities=entities_mem,
                results_images=img_mem,
                results_places=places_mem,
                images_domains=img_mem_domains,
                results_captions=captions,
                captions_domains=captions_domains,
                query_img_regions=qImg if args.consistency == "san" else None,
                qtext_clip=qCap_clip,
                qimage_clip=qImg_clip,
            )
            # compute loss
            # change
            loss = (
                criterion(output, torch.unsqueeze(labels.float(), 1))
                if args.binary_classify
                else criterion(output, labels.flatten())
            )
            total_loss += loss.item()
            # compute correct predictions
            _, pred = torch.max(output.data, 1)
            if args.binary_classify:
                pred = torch.sigmoid(output) >= 0.5
            # truth = torch.unsqueeze(labels,1)
            truth = (
                torch.unsqueeze(labels, 1) >= 0.5 if args.binary_classify else labels
            )
            total_truth += truth.tolist()
            total_pred += pred.tolist()
            # print('pred is ',pred)
            # print('truth is',truth)
            # print('pred is ',pred.shape)
            # print('truth is',truth.shape)

            correct += pred.eq(truth).sum().item()

            total_num += labels.size(0)
            # my weibo
            from collections import Counter

            label_list = labels.view(-1).cpu().numpy().tolist()
            label_ct = Counter(label_list)
            output_list = output.view(-1).cpu().numpy().tolist()
            pred_ct = Counter(output_list)
    f1 = (
        f1_score(total_truth, total_pred)
        if args.binary_classify
        else f1_score(total_truth, total_pred, average="macro")
    )
    avg_loss = total_loss / len(loader)
    acc = (correct / total_num) * 100
    result = classification_report(total_truth,total_pred,digits=4)
    print(result)
    print("acc = ", acc)
    print("f1 =", f1)
    return avg_loss, acc


def train():
    total_loss = 0
    start_time = time.time()
    start_time2 = time.time()
    model.train()
    for (idx, batch) in enumerate(train_dataloader):
        # prepare batches
        print(f'idx {idx}')
        labels = batch[0].cuda()
        entities_mem = batch[1].cuda() if args.use_ent_memory else None
        captions = batch[2].cuda() if args.use_cap_memory else None
        captions_domains = (
            batch[3].cuda() if (args.use_src and args.use_cap_memory) else None
        )
        img_mem = batch[4].cuda() if args.use_img_memory else None
        img_mem_domains = (
            batch[5].cuda()
            if ((args.use_img_memory or args.use_places_memory) and args.use_src)
            else None
        )
        places_mem = batch[6].cuda() if args.use_places_memory else None
        qCap = batch[7].cuda()
        qImg = batch[8].cuda()
        qPlace = batch[9].cuda() if args.use_places_memory else None

        if args.consistency == "clip" and not args.use_clip_for_all:
            qCap_clip = batch[10].cuda()
            qImg_clip = batch[11].cuda()
        else:
            qCap_clip = None
            qImg_clip = None
        # the saved embeddings are the full feature map
        with torch.no_grad():
            img_dim = (
                (model_settings["img_dim_in"] + 1)
                if args.labels_overlap
                else model_settings["img_dim_in"]
            )
            if not args.use_clip_for_all:
                qImg_avg = adaptive_pool(qImg).view(
                    labels.size(0), model_settings["img_dim_in"]
                )
            if img_mem is not None:
                num_img_mem = img_mem.size(1)
                if args.img_rep == "pool" and not args.use_clip_for_all:
                    img_mem = adaptive_pool(img_mem).view(
                        labels.size(0), num_img_mem, img_dim
                    )
                elif args.img_rep == "regions":
                    img_mem = img_mem.view(labels.size(0), num_img_mem, img_dim, 7 * 7)
                    img_mem = img_mem.view(labels.size(0), num_img_mem, 7 * 7, img_dim)
                    img_mem = img_mem.view(labels.size(0), num_img_mem * 7 * 7, img_dim)
                    if img_mem_domains is not None:
                        img_mem_domains = (
                            img_mem_domains.unsqueeze(dim=2)
                            .repeat(1, 1, 49)
                            .view(labels.size(0), num_img_mem * 49)
                        )

        # forward
        output = model(
            qImg_avg if not args.use_clip_for_all else qImg,
            qCap,
            query_places=qPlace,
            entities=entities_mem,
            results_images=img_mem,
            results_places=places_mem,
            images_domains=img_mem_domains,
            results_captions=captions,
            captions_domains=captions_domains,
            query_img_regions=qImg if args.consistency == "san" else None,
            qtext_clip=qCap_clip,
            qimage_clip=qImg_clip,
        )
        # compute loss
        # print(output.shape,labels.shape)
        # change
        loss = (
            criterion(output, torch.unsqueeze(labels.float(), 1))
            if args.binary_classify
            else criterion(output, labels)
        )
        total_loss += loss.item()

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            # print(scheduler.get_last_lr())
        # log
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.9f} | ms/batch {:5.2f} | "
                "loss {:5.2f}".format(
                    epoch,
                    idx,
                    len(train_dataloader),
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"],
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                )
            )
            total_loss = 0
            start_time = time.time()


try:
    # you can exit from training by: ctrl+c
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, val_acc = evaluate(val_dataloader)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | "
            "val acc {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, val_acc
            )
        )
        print("-" * 89)
        log_file_val_loss.write(str(val_loss) + ", " + str(val_acc) + "\n")
        log_file_val_loss.flush()
        if val_loss < stored_loss:
            print("New best model loss")
            stored_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": stored_loss,
                    "best_val_acc": stored_acc,
                },
                os.path.join(args.exp_folder, "best_model_loss.pth.tar"),
            )
        if val_acc > stored_acc:
            print("New best model acc")
            stored_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": stored_loss,
                    "best_val_acc": stored_acc,
                },
                os.path.join(args.exp_folder, "best_model_acc.pth.tar"),
            )
except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")

# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, "best_model_loss.pth.tar")):
    checkpoint = torch.load(os.path.join(args.exp_folder, "best_model_loss.pth.tar"))
    model.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint: '{}')".format(
            os.path.join(args.exp_folder, "best_model_loss.pth.tar")
        )
    )
    # Run on test data.
    test_loss, test_acc = evaluate(test_dataloader)
    print("=" * 89)
    print(
        "| End of training | best loss model | test loss {:5.2f} | test acc {:8.3f}".format(
            test_loss, test_acc
        )
    )
    print("=" * 89)

if os.path.isfile(os.path.join(args.exp_folder, "best_model_acc.pth.tar")):
    checkpoint = torch.load(os.path.join(args.exp_folder, "best_model_acc.pth.tar"))
    model.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint: '{}')".format(
            os.path.join(args.exp_folder, "best_model_acc.pth.tar")
        )
    )
    # Run on test data.
    test_loss, test_acc = evaluate(test_dataloader)
    print("=" * 89)
    print(
        "| End of training | best acc model | test loss {:5.2f} | test acc {:8.3f}".format(
            test_loss, test_acc
        )
    )
    print("=" * 89)
