from tqdm import tqdm
import ujson as json
import numpy as np
import pickle
import os
import pandas as pd
docred_rel2id = None 
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}

def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        new_map = {}
        
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        
        sent_end = len(sents)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end
        
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos

def save_graphs(graphs, base_filename):
    for idx, graph in enumerate(graphs):
        df = pd.DataFrame(graph)
        filename = f"{base_filename}/graph_{idx + 1}.csv"
        df.to_csv(filename, index=False, header=False)
        print(f"Saved {filename}")


def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int, tokenizer = None): 
    ''' Construct pseudo documents from predictions.'''
    
    pos_samples = 0
    neg_samples = 0
    
    sent_grps = []
    pseudo_features = []

    for pred_rel in pred_rels:
        curr_sents = pred_rel["evidence"]
        if len(curr_sents) == 0:
            continue

        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]]) 
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]])

        if len(set(head_sents) & set(curr_sents)) == 0: 
            curr_sents.append(head_sents[0]) 
        if len(set(tail_sents) & set(curr_sents)) == 0:  
            curr_sents.append(tail_sents[0])

        curr_sents = sorted(set(curr_sents)) 
        if curr_sents in sent_grps:
            continue
        sent_grps.append(curr_sents)

        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents] 
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos] 
        new_input_ids = sum(new_input_ids_each, [])
        new_input_ids = tokenizer.build_inputs_with_special_tokens(new_input_ids)
 
        new_sent_pos = []
        prev_len = 0
        for sent in old_sent_pos: 
            curr_sent_pos = (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        curr_entities = []  
        ent_new2old = {}
        new_entity_pos = []

        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                if mention["sent_id"] in curr_sents:
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0] 
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in mention['pos']]
                    curr_pos.append(pos)

            if curr != []:
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                ent_new2old[len(ent_new2old)] = i

        new_hts = []
        new_labels = []
        for h in range(len(curr_entities)):
            for t in range(len(curr_entities)):
                if h != t:
                    new_hts.append([h, t])
                    old_h, old_t = ent_new2old[h], ent_new2old[t]
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])]
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0]
                    pos_samples += 1 - curr_label[0]
        hts_graph = create_hts_graph(new_hts, new_entity_pos)

        pseudo_feature = {'input_ids': new_input_ids,
                    'entity_pos': new_entity_pos,
                    'labels': new_labels,
                    'hts': new_hts,
                    'sent_pos': new_sent_pos,
                    'sent_labels': None,
                    'title': raw_feature['title'],
                    'entity_map': ent_new2old, 
                    'hts_graph': hts_graph
                    }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples


def create_hts_graph(hts, entities):
    N_nodes = len(hts)
    nodes_adj = np.zeros((N_nodes, N_nodes), dtype=np.int32)
    edges_cnt = 1
    for i in range(len(hts)):
        for j in range(i+1, len(hts)):
            ht1 = hts[i]
            ht2 = hts[j]
            if ht1[0] == ht2[1]:
                nodes_adj[i,j] = edges_cnt
                nodes_adj[j,i] = edges_cnt
            elif ht1[1] == ht2[0]:
                nodes_adj[i,j] = edges_cnt
                nodes_adj[j,i] = edges_cnt
    return nodes_adj


def get_chunks(sent_pos, max_seq_length, overlap=2):
    """Split sentence indices into overlapping chunks that fit in max_seq_length tokens."""
    if len(sent_pos) == 0:
        return [[]]
    
    total_tokens = sent_pos[-1][1]
    if total_tokens <= max_seq_length - 2:
        return [list(range(len(sent_pos)))]
    
    chunks = []
    start_sent = 0
    
    while start_sent < len(sent_pos):
        end_sent = start_sent
        while end_sent < len(sent_pos):
            token_count = sent_pos[end_sent][1] - sent_pos[start_sent][0]
            if token_count > max_seq_length - 2:
                break
            end_sent += 1
        
        if end_sent == start_sent:
            end_sent = start_sent + 1
        
        chunks.append(list(range(start_sent, end_sent)))
        
        if end_sent >= len(sent_pos):
            break
        
        start_sent = end_sent - overlap
        if start_sent < 0:
            start_sent = 0
    
    return chunks


def build_chunk_feature(sample, chunk_sent_ids, sents, sent_map, sent_pos, 
                        tokenizer, transformer_type, docred_rel2id):
    """Build a single feature dict for one chunk of sentences."""
    
    entities = sample['vertexSet']
    
    # Remap token positions for this chunk
    chunk_start_token = sent_pos[chunk_sent_ids[0]][0]
    
    # Build chunk sents (tokens)
    chunk_sents = []
    for sid in chunk_sent_ids:
        s, e = sent_pos[sid]
        chunk_sents.extend(sents[s:e])
    
    # Truncate if single sentence is too long
    max_chunk_tokens = 1024 - 2  # reserve for [CLS] [SEP]
    chunk_sents = chunk_sents[:max_chunk_tokens]
    
    # Build chunk sent_pos (relative to chunk start)
    chunk_sent_pos = []
    for sid in chunk_sent_ids:
        s, e = sent_pos[sid]
        rel_s = s - chunk_start_token
        rel_e = e - chunk_start_token
        if rel_s >= len(chunk_sents):
            break
        rel_e = min(rel_e, len(chunk_sents))
        chunk_sent_pos.append((rel_s, rel_e))
    
    chunk_sent_set = set(chunk_sent_ids)
    
    # Find which entities have mentions in this chunk
    # old_ent_idx -> new_ent_idx
    ent_old2new = {}
    chunk_entity_pos = []
    
    for ent_idx, entity in enumerate(entities):
        mentions_in_chunk = []
        for m in entity:
            if m["sent_id"] in chunk_sent_set:
                start = sent_map[m["sent_id"]][m["pos"][0]] - chunk_start_token
                end = sent_map[m["sent_id"]][m["pos"][1]] - chunk_start_token
                if start >= 0 and start < len(chunk_sents):
                    mentions_in_chunk.append((start, end))
        
        if mentions_in_chunk:
            ent_old2new[ent_idx] = len(chunk_entity_pos)
            chunk_entity_pos.append(mentions_in_chunk)
    
    if len(chunk_entity_pos) < 2:
        return None  # need at least 2 entities for pairs
    
    # Build labels and hts for entities in this chunk
    train_triple = {}
    if "labels" in sample:
        for label in sample['labels']:
            h, t = label['h'], label['t']
            if h not in ent_old2new or t not in ent_old2new:
                continue
            new_h = ent_old2new[h]
            new_t = ent_old2new[t]
            r = int(docred_rel2id[label['r']])
            evidence = label['evidence']
            
            if (new_h, new_t) not in train_triple:
                train_triple[(new_h, new_t)] = [{'relation': r, 'evidence': evidence}]
            else:
                train_triple[(new_h, new_t)].append({'relation': r, 'evidence': evidence})
    
    num_entities = len(chunk_entity_pos)
    relations, hts, sent_labels = [], [], []
    doc_rel = [0] * len(docred_rel2id)
    pos_samples = 0
    neg_samples = 0
    
    for h, t in train_triple.keys():
        relation = [0] * len(docred_rel2id)
        sent_evi = [0] * len(chunk_sent_pos)
        
        for mention in train_triple[h, t]:
            relation[mention["relation"]] = 1
            doc_rel[mention["relation"]] = 1
            for evi_sid in mention["evidence"]:
                # Map original sent_id to chunk-local sent index
                if evi_sid in chunk_sent_ids:
                    local_idx = chunk_sent_ids.index(evi_sid)
                    if local_idx < len(sent_evi):
                        sent_evi[local_idx] += 1
        
        relations.append(relation)
        hts.append([h, t])
        sent_labels.append(sent_evi)
        pos_samples += 1
    
    for h in range(num_entities):
        for t in range(num_entities):
            if h != t and [h, t] not in hts:
                relation = [1] + [0] * (len(docred_rel2id) - 1)
                sent_evi = [0] * len(chunk_sent_pos)
                relations.append(relation)
                hts.append([h, t])
                sent_labels.append(sent_evi)
                neg_samples += 1
    
    assert len(relations) == num_entities * (num_entities - 1)
    
    input_ids = tokenizer.convert_tokens_to_ids(chunk_sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    
    hts_graph = create_hts_graph(hts, chunk_entity_pos)
    
    feature = {
        'input_ids': input_ids,
        'entity_pos': chunk_entity_pos,
        'labels': relations,
        'hts': hts,
        'sent_pos': chunk_sent_pos,
        'sent_labels': sent_labels,
        'title': sample['title'],
        'doc_rel': doc_rel,
        'hts_graph': hts_graph
    }
    
    return feature, pos_samples, neg_samples


def read_docred(file_in, 
                tokenizer, 
                transformer_type="bert",
                max_seq_length=1024, 
                teacher_sig_path="",
                single_results=None):

    global docred_rel2id
    if docred_rel2id is None:
        data_dir = os.path.dirname(file_in)
        rel2id_path = os.path.join(data_dir, "rel2id.json")
        docred_rel2id = json.load(open(rel2id_path, 'r'))
        print(f"Loaded rel2id from {rel2id_path} ({len(docred_rel2id)} relations)")

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None

    with open(file_in, "r") as fh:
        data = json.load(fh)

    if teacher_sig_path != "":
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        attns = pickle.load(open(attns_file, 'rb'))

    if single_results != None:  
        pred_pos_samples = 0
        pred_neg_samples = 0
        pred_rels = single_results
        title2preds = {}
        for pred_rel in pred_rels:
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel)
            else:
                title2preds[pred_rel["title"]] = [pred_rel]

    for doc_id in tqdm(range(len(data)), desc="Loading examples"):

        sample = data[doc_id]
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # Get chunks
        chunks = get_chunks(sent_pos, max_seq_length, overlap=2)
        
        for chunk_idx, chunk_sent_ids in enumerate(chunks):
            result = build_chunk_feature(
                sample, chunk_sent_ids, sents, sent_map, sent_pos,
                tokenizer, transformer_type, docred_rel2id
            )
            
            if result is None:
                continue
            
            feature, chunk_pos, chunk_neg = result
            pos_samples += chunk_pos
            neg_samples += chunk_neg
            
            # Modify title for chunks (keep original if single chunk)
            if len(chunks) > 1:
                feature['title'] = f"{sample['title']}_chunk{chunk_idx}"
            
            if teacher_sig_path != '':
                # Teacher attns not supported for chunks
                pass

            if single_results != None:
                offset = 1 if transformer_type in ["bert", "roberta"] else 0
                title = sample["title"]
                if title in title2preds:
                    pseudo_features, p_pos, p_neg = get_pseudo_features(
                        feature, title2preds[title], 
                        # Need to rebuild entities for this chunk
                        [entities[i] for i in range(len(entities))],
                        sent_map, offset, tokenizer
                    )
                    if single_results != None:
                        pred_pos_samples += p_pos
                        pred_neg_samples += p_neg
                    features.extend(pseudo_features)
                    i_line += len(pseudo_features)
                    continue
            
            features.append(feature)
            i_line += 1

    print("# of documents {}.".format(i_line))
    if single_results != None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))
    else:        
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))

    return features