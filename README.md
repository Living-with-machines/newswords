# NewsWords

NewsWords is a toolbox for studying the content of historical British Newspapers during the long nineteenth century.

It enables you to study the distribution of words with high level of granulary and complexity. 

Questions you answers with NewsWords are: Did Liberal newspaper talk more 'workers' about than Conservative titles? When and where did newspaper refer the 'droughts' or other environmental terms?

NewsWords is a toolbox build on two 

## Quick Tour

The most convenient way to explore NewsWords is by downloading the processed count data from Zenodo and unzip `sparse_matrices.zip`.

After following the installation instructions, you open `Explore_Distributed_Corpus.ipynb` and follow the instructions. 

After opening the notebook, you need to set path to the location where you unzipped the data.
```{python}
path = 'path/to/data'
corpus = DistributedCorpus(path,n_cores=12)
```

nd b) the `query_dict`.








## Installation


Code for the counts data derived from historical newspapers


```{bash}
conda create -n newswords python=3.10
```

```{bash}
conda activate newswords
```

```{bash}
conda install -c anaconda ipykernel
```

```{bash}
git clone https://github.com/Living-with-machines/newswords.git
```

```{bash}
cd ./newswords
```

```{bash}
pip install -r requirements.txt
```

## Download


## Process


## Query