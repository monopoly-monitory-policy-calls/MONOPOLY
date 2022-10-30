import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import traceback
from dateutil import parser
from datetime import datetime, timedelta
from numbers_parser import Document

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def exists(path):
    ans = os.path.isfile(path)
    return ans

bank_target_dict = {'scripts_boc': "Canada.numbers",'scripts_boe': "England.numbers",'scripts_bonz': "New Zealand.numbers",'scripts_ecb': "Europe.numbers",'scripts_frb': "US.numbers", 'scripts_bosa': "South Africa.numbers"}

def load_labels_df(path):
    doc = Document(path)
    sheets = doc.sheets()
    tables = sheets[0].tables()
    rows = tables[0].rows(values_only=True)
    df = pd.DataFrame(rows).drop(columns=[0])
    df = df[1:]
    return df

def get_subset(df, target_var=1):
    #df = pd.read_csv(path, skiprows=1, engine='python')
    #df.columns = np.arange(0,len(df.columns))
    #df = df.drop(columns=[0])
    n = target_var
    target_col_idx = 2*n
    date_col_idx = target_col_idx - 1
    df_sub = df[[date_col_idx,target_col_idx]].dropna()
    df_sub = df_sub.rename(columns=df_sub.iloc[0]).drop(df_sub.index[0]).reset_index(drop=True)
    return df_sub

def get_target(df_target, curr_date_parsed, offset, ctry, movement=False):
    target_date = curr_date_parsed + timedelta(days=offset)
    
    if ctry=="US.numbers" or ctry=="Europe.numbers" or ctry=="England.numbers":
        target_date = target_date.strftime("%-m/%-d/%Y")
        curr_date = curr_date_parsed.strftime("%-m/%-d/%Y")
    else:
        target_date = target_date.strftime("%Y-%m-%d")
        curr_date = curr_date_parsed.strftime("%Y-%m-%d")

    if movement:
        row_label_1 = df_target[df_target[df_target.columns[0]]==target_date]
        row_label_2 = df_target[df_target[df_target.columns[0]]==curr_date]

        if len(row_label_1)==0 or len(row_label_2)==0:
            return np.random.randint(0,2), parser.parse(target_date)

        return int(row_label_1.values[0][1] > row_label_2.values[0][1]), parser.parse(target_date)
    else:
        row_label = df_target[df_target[df_target.columns[0]]==target_date]

        if len(row_label)==0:
            return 0, parser.parse(target_date)
        return row_label.values[0][1], parser.parse(target_date)



class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="../", subclip_maxlen=-1):
        
        self.data_path = data_path
        self.sub_data = ['scripts_boc','scripts_bonz','scripts_ecb','scripts_frb', 'scripts_bosa']

        self.text = []
        self.audio = []
        self.video = []
        self.labels = []
        self.timestamps = []
        self.subclip_mask = []
        self.subclip_maxlen = subclip_maxlen

    def load_data(self, data_path=None, offset=1, movement=False):
        errs = 0
        tot = 0
        if data_path is None:
            data_path = self.data_path
        for sub in tqdm(self.sub_data):
            if tot==40:
                break
            print("Loading: ", sub)
            root_folder = os.path.join(data_path, sub)
            csvfile = [each for each in os.listdir(root_folder) if each.endswith('.csv')][0]
            df = pd.read_csv(os.path.join(root_folder, csvfile), header=None)
            root_folder = os.path.join(root_folder, "data")
            ctry = bank_target_dict[sub]
            df_target = load_labels_df(os.path.join(data_path, "bloomberg data", ctry))
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    tot+=1
                    if tot==40:
                        break
                    st = row.values[1]
                    st = eval(st)
                    bank = st['Bank']
                    date = st['Date']
                    video = st['Video_links']
                    folder = date+'_'+bank+'_'+str(idx)
                    folderpath = os.path.join(root_folder, folder)
                    if not os.path.isdir(folderpath):
                        folder = date+'_'+bank
                        folderpath = os.path.join(root_folder, folder)
                    parsed_date = parser.parse(date)
                    labels = []
                    for i in range(6):
                        df_subset = get_subset(df_target, i+1)
                        label, timestamp = get_target(df_subset, parsed_date, offset, ctry, movement)
                        labels.append(label)
                    videofile = os.path.join(folderpath, "video_fragments.npz")
                    #print(videofile)
                    transcriptfile = os.path.join(folderpath, "bert_embeddings.npz")
                    audiofile = os.path.join(folderpath, "wav2vec2_embs.npz")
                    if exists(videofile) and exists(transcriptfile) and exists(audiofile):
                        video_embs = np.load(videofile, allow_pickle=True)['arr_0']
                        if self.subclip_maxlen==-1:
                            video_embs = [np.mean(x, axis=0, keepdims=True) for x in video_embs]
                            self.subclip_maxlen=1
                        s_mask = np.zeros((len(video_embs), self.subclip_maxlen))
                        for idx,vid in enumerate(video_embs):
                            s_mask[idx,:len(vid)] = 1
                        video_embs = [nn.ZeroPad2d((0, 0, 0, self.subclip_maxlen - len(x)))(torch.tensor(x, dtype=torch.float)) for x in video_embs]
                        bert_embs = np.load(transcriptfile, allow_pickle=True)['arr_0']
                        wav2vec2_embs = np.load(audiofile, allow_pickle=True)['arr_0']
                        self.video.append(torch.stack(video_embs))
                        self.subclip_mask.append(s_mask)
                        self.text.append(bert_embs)
                        self.audio.append(wav2vec2_embs)
                        self.labels.append(np.array(labels))
                        self.timestamps.append(datetime.timestamp(timestamp))
                    else:
                        with open('errors_dataloader.txt', 'a') as f:
                            f.write(str(folderpath)+"\n")
                        errs+=1
                except:
                    with open('errors_dataloader.txt', 'a') as f:
                            f.write(str(folderpath)+"\n")
                    errs+=1
                    #print(traceback.format_exc())
            print("ERRORS CURRENT: ", errs)

        print("SKIPPED: ", errs, " out of ", tot)

        #self.labels = list(np.random.rand(len(self.video)))
        #self.labels = self.labels * 200
        #self.text = self.text * 200
        #self.audio = self.audio * 200
        #self.video = list(self.video) * 200
        #self.labels = self.labels * 200
        #self.subclip_mask = self.subclip_mask * 200
        print("DONE.")

    def make_splits(self, ratios = [0.7, 0.1, 0.2]):
        indices = np.argsort(np.array(self.timestamps))
        n = len(indices)
        edges = [0, int(ratios[0]*n), int((ratios[0]+ratios[1])*n), n]
        train_idx = indices[edges[0]:edges[1]]
        val_idx = indices[edges[1]:edges[2]]
        test_idx = indices[edges[2]:edges[3]]
        print(len(train_idx))
        print(len(val_idx))
        print(len(test_idx))
        return train_idx, val_idx, test_idx

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        #label = torch.tensor(self.labels[idx], dtype=torch.float)
        label = self.labels[idx]
        video = self.video[idx]
        audio = torch.tensor(self.audio[idx], dtype=torch.float)
        text = torch.tensor(self.text[idx], dtype=torch.float)
        subclip_mask = torch.tensor(self.subclip_mask[idx], dtype=torch.bool)

        return label, video, audio, text, subclip_mask