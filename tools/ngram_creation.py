#from curses import meta
#from mimetypes import suffix_map
from pathlib import Path
#from attr import validate
from tqdm import tqdm_notebook
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import save_npz, load_npz, vstack
from multiprocessing import Pool
from collections import Counter
import shutil
from tools.helpers import *

class JSONHandler:
    """class for handling and organizing the json outputs"""
    def __init__(self,path_to_json: str):
        """the json files are the monthly ngram outputs for each newspaper """
        self.path_to_json : str = path_to_json
        self.json_paths : list = list(Path(self.path_to_json ).glob('*.json'))
        
        # get all nlp ids in the collection
        self.nlp_ids : list = list(set(
                                map(nlp_id,
                                    self.json_paths)))

        self.shape : tuple = (len(self.json_paths),len(self.nlp_ids))

    def __str__(self):
        return f'{len(self.json_paths)} JSON files for {len(self.nlp_ids)} newspapers'

    def __len__(self):
        return len(self.json_paths)

    def organize_by_nlp(self):
        """organize by json files by nlp, map nlp id to json file path"""
        self.by_nlp = defaultdict(list)
        
        for js in self.json_paths:
            self.by_nlp[nlp_id(js)].append(js)
        
        return True

    def check_json(self):
        def _check_valid_json(path):
            try:
                with open(path) as in_json:
                    json.load(in_json)
            except Exception as e:
                path.unlink()
                print(f"Removing {path} not valid JSON")
                
        json_files = Path(self.path_to_json).glob('*.json')
        for p in tqdm_notebook(json_files):
           _check_valid_json(p)
        print(len(list(json_files)))

        # pool = Pool(processes=4)
        # results = pool.map(_check_valid_json, json_files)

        # with Pool(self.n_cores) as pool:
        #     futures = [pool.apply_async(f) for f in functions]
        #     results = [fut.get() for fut in futures]

class CorpusProcessor:
    """classs for processing all newspapers"""
    def __init__(
                self,
                json_handler: JSONHandler,
                vocab : list, # vocab is required
                save_to : str, # '../sparse_ngrams',
                min_threshold : int = 5,
                #save_to : str = '../sparse_ngrams',
                n_cores : int = 6
                    ):
        
        self.json_handler = json_handler
        self.json_handler.organize_by_nlp()
        self.vocab = vocab

        self.min_threshold = min_threshold
        self.save_to = Path(save_to)
        self.save_to.mkdir(exist_ok=True)

        self.n_cores = n_cores

    # def check_json(self):
    #     def _check_valid_json(path):
    #         try:
    #             with open(path) as in_json:
    #                 json.load(in_json)
    #         except Exception as e:
    #             path.unlink()
    #             print(f"Removing {path} not valid JSON")
                
    #     json_files = Path(self.json_handler.path_to_json).glob('*.json')
    #     for p in tqdm_notebook(json_files):
    #         _check_valid_json(p)
        #print(len(list(json_files)))

    def process_ngrams(self: int):
        """convert convert to sparse matrix"""
        
        functions = [NewspaperProcessor(nlp,
                                    self.json_handler,
                                    self.vocab,
                                    self.save_to,
                                    min_threshold=self.min_threshold
                                        )._process_by_nlp for nlp in self.json_handler.nlp_ids
                                        ]
        with Pool(self.n_cores) as pool:
            futures = [pool.apply_async(f) for f in functions]
            results = [fut.get() for fut in futures]

    def merge_metadata(self,out_path,totals,
                    npd_links_path=None,
                    npd_data_path=None):

        """warning: updating metadata doesn't seem to work
        TO DO: why?
        """
        
        out_path.mkdir(exist_ok=True)
        metadata = []
        for nlp in tqdm_notebook(self.json_handler.nlp_ids):
            m = pd.read_csv(f'{self.save_to}/{nlp}_metadata.csv',index_col=0)
            m['NLP'] = nlp
            metadata.append(m)
       
        metadata = pd.concat(metadata,axis=0)
        metadata['totals'] = totals
        
        
        if npd_links_path:
            
            metadata['idx'] = list(range(metadata.shape[0]))
            metadata['link'] = metadata.apply(lambda x: f'{x.NLP}_{x.year}',axis=1)
            npd_links = pd.read_csv(npd_links_path,index_col=0,dtype={'NLP':str,'AcquiredYears':int})
            npd_links['link'] = npd_links.apply(lambda x: f'{x.NLP.zfill(7)}_{x.AcquiredYears}',axis=1)
            print(metadata['link'][:10],npd_links['link'][:10])
            # !!!! TO DO: this join adds rows, try to find out why
            # some newspaper appear in multiple collections
            # the line below should fix it but not sure if it's correct
            npd_links = npd_links.drop_duplicates(subset='link')
            metadata = metadata.merge(npd_links,
                                        right_on='link',
                                        left_on='link',
                                        how='left',
                                        suffixes=['_orig',''],
                                        validate='many_to_one')
            
           
        if npd_data_path:
            npd_data = pd.read_csv(npd_data_path)
            metadata  = metadata.merge(npd_data,left_on='link_to_mpd',right_on='id',how='left',suffixes=['','_npd'])
            
        
        print(f'Saving metadata of size {metadata.shape}')
        metadata.to_csv(out_path / 'metadata.csv')
        #return metadata
        
        

    def merge_sparse_matrices(self,out_path: str, override:bool=False, **kwargs):
        if isinstance(out_path,str):
            out_path = Path(out_path)
        out_path.mkdir(exist_ok=True)
        data = []

        if (out_path / 'sparse_matrix.npz').is_file() and not override:
            
            print('Sparse matrix created...')
            data = load_npz(out_path / 'sparse_matrix.npz')

        else:
            for nlp in tqdm_notebook(self.json_handler.nlp_ids):
                matrix = load_npz(f'{self.save_to}/{nlp}_sparse_matrix.npz').astype(np.int32)
                #print(matrix.shape)
                data.append(matrix)
            
            data = vstack(data)
            print(f'Saving sparse matrix of size {data.shape}')
            save_npz(out_path / 'sparse_matrix.npz', data, compressed=True)
        
        print('Copying vocab files')
        
        for f in ['vocab.json','mapping.json']:
            shutil.copy(self.save_to / f, out_path / f)
        
        totals = data.sum(axis=1)
  
        self.merge_metadata(out_path,totals,**kwargs)
       
        print('Done.')
        
        return True


class Vocab(CorpusProcessor):
    """class for computing a vocabulary over all newspapers"""
    def __init__(self, json_handler,**kwargs):
        CorpusProcessor.__init__(self,json_handler,[],**kwargs)
        self.wc_by_nlp = None
        self.vocab = list()
    
    def __len__(self):
        return len(self.vocab)
    
    def filter_dict(self, dictionary: dict) -> dict:
        """remove keys based on frequency"""
        return dict(filter(lambda x: x[1] > self.min_threshold, dictionary.items()))

    def add_dicts_from_path(self,items: list) -> Counter:
        """combine json files into into one Counter object"""
        
        count_total = Counter()
        
        try:
            for item in items:
                
                with open(item) as in_json:
                    count = self.filter_dict(Counter(json.load(in_json)))
                count_total += count
        
        except Exception as e:
            print(e)
        
        return count_total
    
    def nlp_counts(self):
        """gather counts by newspaper nlp"""
        pool = Pool(self.n_cores)
        
        self.wc_by_nlp = list(tqdm_notebook(pool.imap(self.add_dicts_from_path, 
                                                    [paths for paths in list(self.json_handler.by_nlp.values())]
                                                        )
                                                    )
                                                )
        pool.close()
        pool.join()
        
    def total_counts(self,check_exist=False):
        """combine counts by nlp into one word frequency mapping"""
        self.wc_total = Counter()
        
        if check_exist:
            self.load()

        if not self.wc_by_nlp:
            self.wc_by_nlp = self.nlp_counts()
        
        for wc in tqdm_notebook(self.wc_by_nlp):
            wc_f = self.filter_dict(wc)
            self.wc_total += wc_f
        
        self.vocab = list(self.wc_total.keys())

    def filter_by_min_threshold(self,min_counts=100):
        self.wc_total = Counter(dict(filter(lambda x: x[1] >= min_counts,self.wc_total.items())))
        self.vocab = list(self.wc_total.keys())
    
    def filter_by_lambda(self,lamdba_func):
        self.wc_total = Counter(dict(filter(lamdba_func,self.wc_total.items())))
        self.vocab = list(self.wc_total.keys())

    def save(self):
        """save counts and vocab"""
        if self.wc_by_nlp:
            with open(self.save_to / 'word_counts_by_nlp.json','w') as out_json:
                json.dump(self.wc_by_nlp, out_json)

        if self.wc_total:
            with open(self.save_to / 'word_counts.json','w') as out_json:
                json.dump(self.wc_total, out_json)
        
            with open(self.save_to / 'vocab.json','w') as out_json:
                json.dump(self.vocab, out_json)

            vectorizer = DictVectorizer()
            vectorizer.fit([{w:0 for w in self.vocab}])
            
            with open(self.save_to / 'mapping.json','w') as out_json:
                json.dump(vectorizer.vocabulary_, out_json)

    def load(self):
        """load counts"""
        if (self.save_to / 'word_counts_by_nlp.json').is_file():
            with open(self.save_to / 'word_counts_by_nlp.json','r') as in_json:
                self.wc_by_nlp = json.load(in_json)

        if (self.save_to / 'word_counts.json').is_file():
            with open(self.save_to / 'word_counts.json','w') as in_json:
                self.wc_total = json.load(in_json)

        if (self.save_to / 'vocab.json').is_file():
            with open(self.save_to / 'vocab.json','w') as in_json:
                self.vocab = json.load(in_json)


class NewspaperProcessor(CorpusProcessor):
    """class for processing ngrams by newspaper"""
    def __init__(self, 
                nlp: str, 
                json_handler : JSONHandler,
                vocab: list,
                save_to: str,
                **kwargs,
                ):
        self.nlp = nlp
        CorpusProcessor.__init__(self,json_handler,vocab,save_to,**kwargs)
        
        
    def __str__(self):
        return f'<Processing newspaper with NLP {self.nlp}.'
    
    def check_done(self):
        """check if newspaper has already been processed"""
        if (self.save_to / f'{self.nlp}_metadata.csv').is_file():
            return True
        return False
    
    def _load_by_nlp(self):
        self._files = (json.load(open(js,'r')) for js in self.json_handler.by_nlp[self.nlp])

    def _generate_metadata(self) -> pd.DataFrame:
        """obtain metadata for rows in the sparse matrix
        for now we only have date
        --> TO DO: include other relevant metadata
        """

        data = []
        for i,js in enumerate(self.json_handler.by_nlp[self.nlp]):
            year = js.name.split('_')[1]
            year_month = '-'.join([js.name.split('_')[1],js.name.split('_')[2][:-5]])
            #politics = meta_dict.get(int(js.name.split('_')[0]),['none'])[0]
            #place = meta_dict.get(int(js.name.split('_')[0]),['none'])[-1]
            data.append([i,year,year_month])
        return pd.DataFrame(data,columns=['idx','year','month'])

    def _process_by_nlp(self):
        """process a newspaper with a specific nlp"""

        vectorizer = DictVectorizer()
        print(f'Processing {self.nlp}')
        
        if self.check_done():
            print(f'Processing {self.nlp} already completed.')
            return 

        self._load_by_nlp() # get file paths
        # Changing code here
        # We will require a stable vocab for the collection
        #if self.vocab:
        vectorizer.fit([{w:0 for w in self.vocab}])
        X = vectorizer.transform(self._files)
            
        #else:
        #    X_t = vectorizer.fit_transform(self._files)
        #    word_totals = X_t.sum(axis=0)
        #    include = np.where(word_totals >= self.min_threshold)[1]
        #
        #    X = X_t[:,include]
        #    vocab = list(np.array(vectorizer.get_feature_names())[include])
        print(self.save_to)
        print(self.save_to / f'{self.nlp}_sparse_matrix.npz')
        save_npz(self.save_to / f'{self.nlp}_sparse_matrix.npz', X, compressed=False)
        
        df = self._generate_metadata()
        df.to_csv(self.save_to / f'{self.nlp}_metadata.csv')
        
        #if not self.vocab: # only write vocab if it does not already exist
        #    json.dump(vocab,open(self.save_to / f'{self.nlp}_sparse_matrix_columns.json','w'))
        
        print(f'Processing {self.nlp} done.')