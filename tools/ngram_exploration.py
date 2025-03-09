from sqlite3 import Timestamp
from tqdm import tqdm_notebook
from scipy.sparse import vstack,load_npz
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
from scipy.sparse import save_npz
from tools.helpers import *
import pandas as pd
import numpy as np
import json
#import gc3 




def _obtain_metadata(path,nlp,query_dict):
    newspaper = Newspaper.load(path,nlp)
    newspaper.query_by_idx(query_dict)
    return newspaper.metadata

def _obtain_timeseries(path,nlp,window):
    newspaper = Newspaper.load(path,nlp)
    return newspaper.entropy_timeseries(window)

class Corpus:
    def __init__(self,counts,metadata,vocab):
        self.counts = counts
        self.metadata = metadata
        self.vocab = vocab
        self.vocab_set = set(vocab.keys())

    @classmethod
    def load(cls,path):
        if isinstance(path,str):
            path = Path(path)
        
        counts = load_npz(path / 'sparse_matrix.npz')
        metadata = pd.read_csv(path / 'metadata.csv')
        vocab = json.load(open(path / 'mapping.json','rb'))

        return cls(counts,metadata,vocab)


    def save(self,path):
        if isinstance(path,str):
            path = Path(path)
        
        path.mkdir(exist_ok=True)

        save_npz( path / 'sparse_matrix.npz', self.counts , compressed=False)
        self.metadata.to_csv(path / 'metadata.csv')
        with open(path / 'mapping.json','w') as out_json:
                json.dump(self.vocab, out_json)
    

        return True

    def simplify_political_labels(self, classification_csv, target_col, new_col='Leaning'):
        pol_labels = pd.read_csv(classification_csv, index_col=0)
        self.convert_dict = pol_labels[['label_orig',target_col]].set_index('label_orig',drop=True).to_dict()[target_col]
        self.metadata[new_col] = self.metadata['S-POL'].replace(self.convert_dict).astype(str)


    def reduce_corpus(self,pmdict):
        
        print('Original count matrix: ',self.counts.shape)
        print('Original metadata: ',self.metadata.shape)
        metadata_sel = self.metadata[self.metadata.year.isin(pmdict['year_range'])]

        row_idx = metadata_sel.idx.values
        self.counts = self.counts[row_idx]#

        self.metadata = metadata_sel.reset_index()
        self.metadata['idx'] = list(self.metadata.index)
        print('Reduced metadata: ',self.metadata.shape)

        print('Reduced count table after metadata filtering : ',self.counts.shape)

        word_counts = np.array(self.counts.sum(axis=0)).flatten()

        words = np.array(list(self.vocab.keys()))
        max_freq = sorted(word_counts, reverse=True)[pmdict['remove_n_most_frequent']]
        col_idx = list(np.where((word_counts > pmdict['min_freq']) & (word_counts < max_freq) )[0])

        # mapping_inv maps column index to token
        mapping_inv = {i: w for w, i in self.vocab.items()}
                
        col_idx = [v for v in col_idx if mapping_inv[v].isalpha() # keep alphabetic words   
                and len(mapping_inv[v]) >= pmdict['word_length'] # keep words longer than three characters
                #and len(set(mapping_inv[v])) != 1 # keep workds that have more than one unique character, i.e. remove aaa etc
                ] 

        print('Number of words in vocabulary after filtering: ',len(col_idx))

        # map token related to old index n to new position
        mapping_red = {mapping_inv[n]:i for i, n in enumerate(col_idx)}
                
        # set the updated mapping
        self.vocab = mapping_red
        self.vocab_set = set(self.vocab.keys())
        self.counts = self.counts[:,col_idx]

        print('Reduced count matrix after metadata and word frequency filtering: ',self.counts.shape)
        
    
    def refresh_metadata(self,path):
        self.metadata = pd.read_csv(Path(path) / 'metadata.csv')

    def query(self,queries=None):
        query_ids = [self.vocab[q] for q in queries if q in self.vocab_set]
        
        if not query_ids:
            print('Could not find tokens in corpus')
            return None
        if len(queries) > 1:
            res = self.counts[:,query_ids].sum(axis=1)
        else:
            res = self.counts[:,query_ids].todense()
        
        return np.asarray(res).reshape(-1)

    
    def complete_records(self, target_col:str) -> pd.DataFrame:
        """function to add information to the press directories detaframe. we use the 
        chain_id column to add missing information, for example if, for one year, a 
        newspaper doesn't have a S-POL we insert one that is closest in time from a row with
        the same chain_id"""

        print('Before: ',sum(~self.metadata[target_col].isnull()))
        
        df = self.metadata.copy()
        df[f'value_{target_col}_source_idx'] = None
        df_chain = df[~df.chain_id.isnull()]
        chains = set(df_chain.chain_id.unique())
    
        for chain_id in chains:
            # get elements for a specific chain_id
            chain_df = df[df.chain_id==chain_id]
            # find those the have NaA values
            no_attr = list(np.where(chain_df[target_col].isnull())[0])

            # if there are empty but not all values are empty
            if no_attr and not (len(no_attr) == chain_df.shape[0]): 
                # look which cells have a value for this columns       
                has_attr = np.where(~chain_df[target_col].isnull())[0]
                # find indices for cells (that have content) closest to an empty cell
                replace_with = chain_df.iloc[[has_attr[np.argmin(abs(has_attr - i))] for i in no_attr]][target_col]
                zipped = list(zip(chain_df.iloc[no_attr].index,replace_with.index,replace_with.values))
                for target_idx, source_idx, cat in zipped:
                    df.loc[target_idx,target_col] = cat
                    df.loc[target_idx,f'value_{target_col}_source_idx'] = source_idx

        self.metadata = df

        print('After: ',sum(~self.metadata[target_col].isnull()))
        


class DistributedCorpus(Corpus):
    def __init__(self,path,n_cores=4):
        self.path = Path(path)
        self.nlps = list(map(nlp_id,self.path.glob('[!.|!metadata.csv]*.csv')))
        
        with open(self.path / 'mapping.json','r') as in_json:
            self.vocab_mapping = json.load(in_json)
        
        self.vocab_set = set(self.vocab_mapping.keys())
        self.shape = (len(self.nlps),len(self.vocab_mapping))
        self.n_cores = n_cores
        self.metadata = pd.read_csv(self.path / 'metadata.csv')
        self.corpus = None

    def plot_vocab_distribution(self,topn=(0,1000), relative=True):
        with open(self.path / 'word_counts.json','r') as in_json:
            wf = json.load(in_json)
        
        t = np.sum(list(wf.values()))
        
        wf_red = dict(sorted(wf.items(), key=lambda x: x[1], reverse=True)[topn[0]:topn[1]])
        
        if relative:
            wf_red = { w: f / t for w,f in wf_red.items()}
            pd.Series(wf_red).plot(figsize=(15,5))
        else:
            pd.Series(wf_red).plot(figsize=(15,5))
        return wf

    def _filter_vocab(self,min_freq: float = .0,max_freq: float=1.0):
        with open(self.path / 'word_counts.json','r') as in_json:
            wf = json.load(in_json)
        
        t = np.sum(list(wf.values()))
        len_old_mapping = len(self.vocab_mapping)
        
        wf_red = dict(filter(lambda x: ((x[1] / t) >= min_freq) and ((x[1] / t) <= max_freq), wf.items()))
        
        self.vocab_mapping = {w: self.vocab_mapping[w] for w in wf_red}
        
        len_mapping = len(self.vocab_mapping)
        self.shape = (len(self.nlps),len(self.vocab_mapping))
        print(f'removed {len_old_mapping - len_mapping} tokens, {round((len_old_mapping-len_mapping)/len_old_mapping,3)} % of the total')
        
    def filter_counts(self,min_freq: float = .0,max_freq: float=1.0):
       
        self._filter_vocab(min_freq, max_freq)
        idx = list(self.vocab_mapping.values())
        
        return self.counts[:,idx]

    def _create_query_dict(self,query_dict):
        """read the query dict and get positional indexes for query terms"""
        query_dict_by_idx = {}
        
        for query_id, query_terms in query_dict.items():
            query_dict_by_idx[query_id] = [self.vocab_mapping[q] for q in query_terms if q in self.vocab_set]
        
        return query_dict_by_idx
            
    def query(self,query_dict):
        
        query_dict = self._create_query_dict(query_dict)
        
        vars = [(self.path,nlp, query_dict) for nlp in self.nlps]
        
        with Pool(self.n_cores) as pool:
            result = pool.starmap(_obtain_metadata, vars)
        
        return result
    
    def add_context(self,result):
        df = pd.concat(result)
        df['link'] = df.apply(lambda x: f'{x.nlp}_{x.month}', axis=1)
        self.metadata['link'] = self.metadata.apply(lambda x: f'{str(x.NLP).zfill(7)}_{x.month}', axis=1)
        num_counts = len([c for c in df.columns if c.startswith('counts_')])
        for i in range(num_counts):
            df['relative_counts_'+str(i+1)] = df['counts_'+str(i+1)] / df['totals']
        return self.metadata.merge(df[['idx','link']+[c for c in df.columns if 'count' in c]], on='link', how='left')

    def counts_by_timestep(self,timestep):
        #if timestep == 'year':
        #    ts = list(range(1780,1920))
        #    
        #elif timestep == 'month':
        #    ts = [f'{i}-{str(j).zfill(2)}'for i in range(1780,1920) for j in range(1,13)]
        ts = self.create_timestep_index(timestep)
        counts = np.zeros((len(ts),len(self.vocab_mapping)),dtype=np.int32)
        for nlp in tqdm_notebook(self.nlps):
            newspaper = Newspaper.load(self.path,nlp)
           
            counts = newspaper.counts_by_timestep(counts, ts, timestep)
        np.save(self.path / f"counts_by_{timestep}.npy", counts)
        return counts,ts
    
    def create_timestep_index(self,timestep):
        if timestep == 'year':
            ts = list(range(1780,1920)) 
        elif timestep == 'month':
            ts = [f'{i}-{str(j).zfill(2)}'for i in range(1780,1920) for j in range(1,13)]
        else:
            ts = []
        return ts

    def load_counts_by_timestep(self,timestep):
        path = self.path / f"counts_by_{timestep}.npy"
        if not path.is_file():
            print(f'File does not exist. Apply self.counts_by_timestep({timestep}) to create count table.')
            return False
        counts = np.load()
        ts = self.create_timestep_index(timestep)
        return counts,ts
    
    def entropy_timeseries(self,window):
      
        vars = [(self.path,nlp, window) for nlp in self.nlps]
        
        with Pool(self.n_cores) as pool:
            result = pool.starmap(_obtain_timeseries, vars)
        
        return result

    def __str__(self):
        return f'< Ngrams corpus with {len(self.nlps)} newspapers and {len(self.vocab)} word types >'

    def __len__(self):
        return len(self.nlps)
    
class Newspaper:
    def __init__(self,nlp,counts=None,metadata=None):
        self.nlp = nlp
        self.counts = counts
        self.metadata = metadata

    @classmethod
    def load(cls,path,nlp):
        counts = load_npz(path / f'{nlp}_sparse_matrix.npz').astype(np.int64)
        metadata = pd.read_csv(path / f'{nlp}_metadata.csv',index_col=0)
        return cls(nlp,counts,metadata)
    
    def query_by_idx(self,query_dict):
        self.metadata['nlp'] = self.nlp # get track of origin of counts
        for q_name, q_idx in query_dict.items():
            
            if q_idx:
                self.metadata[f'counts_{q_name}'] = np.squeeze(np.asarray(self.counts[:,q_idx].sum(axis=1)))
            else: # if none of the query terms appear add a column with zero counts
                self.metadata[f'counts_{q_name}'] = np.array([0] * self.counts.shape[0])
    
        self.metadata['totals'] = np.squeeze(np.asarray(self.counts.sum(axis=1)))
    
    def counts_by_timestep(self,counts,ts,timestep): 
        
        by_timestep = self.metadata.groupby([timestep])['idx'].apply(list).reset_index().sort_values(timestep)
        
        for i,row in by_timestep.iterrows():
            year = int(str(row[timestep]).split('-')[0])
            if year >= 1780 and year < 1920:
                ts_counts = np.squeeze(np.asarray(self.counts[row.idx].sum(axis=0)))
                counts[ts.index(row[timestep])] += ts_counts
                
        return counts

    def entropy_timeline(self,idx,matrix,window):
        past_entropies = [jensenshannon(matrix[idx], matrix[idx-i]) 
                                            for i in range(1,window+1)][::-1]
        future_entropies = [jensenshannon(matrix[idx], matrix[idx+i]) 
                                            for i in range(1,window+1)]
        entropies = np.array(past_entropies + future_entropies)
        return (entropies - np.mean(entropies)) / np.std(entropies) 

    def entropy_timeseries(self,window=6):
        if len(self.metadata) <= (window*2)+1: return []
        
        matrix = np.asarray((self.counts.T / np.squeeze(np.asarray(self.counts.sum(axis=1)))).T)
        
        data = [(self.metadata.month[j],
                self.nlp,
                *self.entropy_timeline(j,matrix,window))
                    for j in range(window,len(self.metadata.month) - window)]
        df = pd.DataFrame(data, columns=['timestamp','nlp',*range(window*2)])
        
        return df

    def __str__(self):
        return f'< Newspaper {self.nlp} with {self.counts.shape[1]} word types and {self.counts.shape[0]} timestamps >'