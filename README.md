# NewsWords

NewsWords is a toolbox for studying the content of historical British Newspapers during the "long" nineteenth century (1780-1920).

NewsWords records word count information by month and is linked to enriched metadata derived from historical Press Directories.

Questions you can answer with NewsWords are: Did Liberal newspaper talk more about 'workers' than Conservative titles? When and where did newspapers mention 'droughts' (See animation below)? 

![](figures/drought_2.gif)
Figure 1: Example of NewsWords output showing the frequency of the word "drought" by place of publication over time

NewsWords is a toolbox built on two datasets:
- Word Count Data: records word frequency by month for each digitised newspaper title between 1780 and 1920.
- Structured Press Directories: provides elaborate descriptions of the newspaper titles, capturing attributes such as politics, price and more.

By combining these sources, NewsWords enables you to study the distribution of words with a high level of granularity and complexity. 

## Quick Tour

The most convenient way to explore NewsWords is by downloading the processed count data from [Zenodo](https://zenodo.org/uploads/14996278) and unzip the `sparse_matrices.zip` file.

After following the installation instructions (see below), you can open `Explore_Distributed_Corpus.ipynb` and follow the instructions. 

You need to change the path to where you unzipped the data.

```python
path = 'path/to/data'
corpus = DistributedCorpus(path,n_cores=12)
```

Next, you can define the `query_dict`. This is a Python dictionary, which maps an identifier (int) to a list of query terms (list[str]). 

`query_dict` defines all the queries you want to run over the NewsWords corpus. In the example below, the first query will capture the frequency of the word 'machine'. The second query counts the words 'train' and 'railway'. Because it takes quite some time to iterate over the complete NewsWords dataset (wich amounts to 85GB unzipped), it is more efficient to define all query terms in advance.

```python
# define the queries you want to run
query_dict =  {1:['machine'], 
               2:['train','railway'],
               3:['accident','accidents'],
               4:['workers','worker'],
               5:['drought','droughts']}
```               

Now, you can run the query using the following line of code.

```python
response = corpus.query(query_dict) # query the word count data
```

The `response` variable will contain a list of Pandas DataFames, one for each newspaper title. Please note that we record the frequency by month. 

The `add_context` method associates each observation with more metadata. This will add more columns with information about the political leaning, price and other attributes that are crucial to understanding the profile and position of each newspaper title.

```python
df = corpus.add_context(res) # add context to the query results
```

The resulting dataframe `df` contains
- the absolute and relative word counts for each query in `query_dict`.
- the enriched metadata for each title.

This allows you to study the distribution of words in the press in novel ways. For example, the image below shows code for plotting timelines showing the relative frequency of the words 'worker(s)' in the Liberal and Conservative press.

Firstly you define the parameters of your search results.

```python
timestep = 'year' # define the time step, the temporal unit of analysis, mostly year
facet = 'Leaning'# define the facet 'Leaning'
facet_values = ['conservative','liberal'] # which labels to use as facet values
start_at, end_at = 1830,1920 # set the date range for the timeline
```

Secondly, you can plot these results with the `seaborn` library. 

```python
plt.figure(figsize=(10,6))
plt.xticks(rotation=90)
plt.title(f'frequency of {str(query_dict[query_dict_key])}')
sns.set(font_scale= 1.25)
sns.lineplot(x=timestep,y=f'relative_counts_{query_dict_key}',data=df[time_filter])
```

Which generates the following lineplot.

![lineplot](figures/lineplot1.png)

## Installation

Create a new environment with the name newswords.

```bash
conda create -n newswords python=3.10
```

Active the newswords environment.

```bash
conda activate newswords
```

Install the ipykernel.

```bash
conda install -c anaconda ipykernel
```

Clone the newswords code and 

```bash
git clone https://github.com/Living-with-machines/newswords.git
cd ./newswords
```

Install all the necessary packages in the requirements file.

```bash
pip install -r requirements.txt
```


