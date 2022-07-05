#!/usr/bin/env python
# coding: utf-8

# # SpatialDE Analysis

# In[1]:


import numpy as np
import pandas as pd
import NaiveDE
import SpatialDE
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm


# ## Import Datasets

# In[2]:


def T_quality(x):
    return np.clip(1-np.log(1+x)/3.9,0,1)


Q_th = 2
min_count = 500
barcodes_df = []
tagList_df = pd.read_csv("../data/tagList_84-gene.csv", sep = ",", usecols = [0,1], header = None, names = ["Seq","Gene"])
datasets = ['170315_161220_4_1','161230_161220_3_1']
for sample in datasets:
    df = pd.read_csv("../data/results/"+sample+"/barcodes.csv", sep = ",")
    df.seq_quality_min=df.seq_quality_min*df.max_dist.apply(T_quality)
    # Add gene names to dataframe
    d = pd.Series(tagList_df.Gene.values,index=tagList_df.Seq).to_dict()
    df["Gene"] = df['letters'].map(d)
    # Downsample barcode coordinate space by factor 8 for easier visualization
    df["global_X_pos"]=df.global_X_pos/8
    df["global_Y_pos"]=df.global_Y_pos/8
    # Remove reads not in the codebook
    df = df.dropna()
    # Filter reads by quality
    df = df[df.seq_quality_min>Q_th]
    # Filter reads by min count per gene
    df["count"] = 0
    for i,row in tagList_df.iterrows():
        df.loc[df["Gene"] == tagList_df.Gene[i],["count"]] = len(df[df["Gene"] == tagList_df.Gene[i]])
    df = df[df["count"]>min_count]
    
    barcodes_df.append(df)


# ## Generate Expression Tables

# In[3]:


# Import and downsample by factor 8 image shape
img_shape = np.round(np.array([[22508, 33566],[22563, 31782]])/8).astype(np.uint)

# Create gene expression table
expression_df = []
for s_idx, df in enumerate(barcodes_df):
    x_min = 0; x_max= img_shape[s_idx,1];
    y_min = 0; y_max= img_shape[s_idx,0];
    batch_size_px=64
    overlap = 8

    express_table = pd.DataFrame(data={}, columns=df.Gene.unique(), index=list((str(x)+"x"+str(y)) for x in range(x_min,x_max,batch_size_px) for y in range(y_min,y_max,batch_size_px))) 
    for i in tqdm(range(x_min,x_max,batch_size_px)):
        for j in range(y_min,y_max,batch_size_px):
            batch_df=df[(df.global_X_pos>=i-(batch_size_px/2)-overlap) & (df.global_X_pos<i+(batch_size_px/2)+overlap) & (df.global_Y_pos>=j-(batch_size_px/2)-overlap) & (df.global_Y_pos<j+(batch_size_px/2)+overlap)]
            if len(batch_df):
                batch_counts = batch_df['Gene'].value_counts()
                express_table.loc[str(i)+'x'+str(j),batch_counts.index]=batch_counts
                
    express_table = express_table.fillna(0)
    
    expression_df.append(express_table)


# In[4]:


# save dataframes
for i,dataset in enumerate(datasets):
    expression_df[i].to_pickle('../data/results/'+dataset+'/SpatialDE_express_table.hdf5')


# In[5]:


# load dataframes
img_shape = np.round(np.array([[22508, 33566],[22563, 31782]])/8).astype(np.uint)
expression_df = []
sample_df = []
for i,dataset in enumerate(datasets):
    plt.rcParams["figure.dpi"] = 150
    plt.subplot(1,2,i+1)    
    
    x_min = 0; x_max= img_shape[i,1];
    y_min = 0; y_max= img_shape[i,0];
    batch_size_px=16
    overlap = 16
    express_table = pd.read_pickle('../data/results/'+dataset+'/SpatialDE_express_table.hdf5')
    # Create sample_info
    sample_info = pd.DataFrame(data={'x':list(x for x in range(x_min,x_max,batch_size_px) for y in range(y_min,y_max,batch_size_px)), 'y':list(y for x in range(x_min,x_max,batch_size_px) for y in range(y_min,y_max,batch_size_px))}, index=list((str(x)+"x"+str(y)) for x in range(x_min,x_max,batch_size_px) for y in range(y_min,y_max,batch_size_px)))
    sample_info['total_counts'] = express_table.sum(axis=1)
    # Dropping empty batches
    express_table = express_table[sample_info.total_counts>10]
    sample_info = sample_info[sample_info.total_counts>10]
    
    expression_df.append(express_table)
    expression_df[i] = expression_df[i].rename(('{}_'+str(i)).format)

    sample_df.append(sample_info)
    sample_df[i] = sample_df[i].rename(('{}_'+str(i)).format)
    sample_df[i]['s'] = i
    plt.scatter(sample_df[i]['x'], sample_df[i]['y'], c=sample_df[i]['total_counts'],s=2);
    plt.axis('equal');


# ## Normalize Gene Expression Tables

# In[6]:


expression_df=pd.concat(expression_df,sort=True)
expression_df=expression_df.dropna(axis=1)
#expression_df=expression_df.fillna(0)
sample_df=pd.concat(sample_df,sort=True)
                    
# Linear regression to account for library size and sequencing depth bias of each patch of gene expression
norm_expr = pd.concat([NaiveDE.stabilize(expression_df[sample_df.s==0].T).T,NaiveDE.stabilize(expression_df[sample_df.s==1].T).T])
resid_expr = pd.concat([NaiveDE.regress_out(sample_df[sample_df.s==0], norm_expr[sample_df.s==0].T, 'np.log(total_counts)').T,NaiveDE.regress_out(sample_df[sample_df.s==1], norm_expr[sample_df.s==1].T, 'np.log(total_counts)').T])
idx = resid_expr.var().sort_values(ascending=False).index


# ## SpatialDE significance test 

# In[7]:


results = []
for i,df in enumerate(datasets):
    X = sample_df.loc[sample_df.s==i,['x', 'y']]
    results.append(SpatialDE.run(X, resid_expr.loc[sample_df[sample_df.s==i].index,:]))


# In[8]:


from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
display_side_by_side(results[0].sort_values('qval').head(10)[['g', 'l', 'qval']],results[1].sort_values('qval').head(10)[['g', 'l', 'qval']])


# In[9]:


for i,df in enumerate(datasets):
    plt.subplot(1,2,i+1)
    plt.scatter(results[i]['FSV'], results[i]['qval'], c='black')
    plt.xlabel('Fraction spatial variance')
    plt.ylabel('Adj. P-value');


# ## Automatic expression histology

# In[10]:


res = []
n_patterns = 20
for i,df in enumerate(datasets):
    sign_results = results[i].query('qval < 0.05')
    X = sample_df.loc[sample_df.s==i,['x', 'y']]
    histology_results, patterns = SpatialDE.aeh.spatial_patterns(X, resid_expr.loc[sample_df[sample_df.s==i].index,:], sign_results, C=n_patterns, l=results[i].l.mean(), verbosity=1)
    res.append({'aeh': histology_results, 'patterns':patterns})


# In[11]:


# Save results
for s,df in enumerate(datasets):
    pickle.dump(res[s], open( "../data/results/"+df+"/SpatialDE_res.hdf5", "wb" ) )


# In[12]:


# Plot Histological Patterns
res = []
n_patterns = 20
for s,df in enumerate(datasets):
    res.append(pickle.load(open( "../data/results/"+df+"/SpatialDE_res.hdf5", "rb")))
    histology_results = res[s]['aeh']
    patterns = res[s]['patterns']
    plt.figure(figsize=(20,10))
    plt.suptitle(df,fontsize=20)
    j=1
    for i in range(n_patterns):
        if len(histology_results[histology_results.pattern==i])>0:
            plt.subplot(4, 5, j)
            plt.scatter(sample_df.loc[sample_df.s==s,'x'], sample_df.loc[sample_df.s==s,'y'], c=patterns[i], s=5,cmap='jet');
            plt.axis('scaled')
            plt.title('Pattern {} - {} genes'.format(i, histology_results.query('pattern == @i').shape[0]), fontsize=10)
            plt.colorbar(ticks=[]);
            plt.xticks([])
            plt.yticks([]);
            j = j+1


# In[13]:


for i,df in enumerate(datasets):
    histology_results = res[i]['aeh']
    print(df)
    for i in histology_results.sort_values('pattern').pattern.unique():
        print('Pattern {}: '.format(i))
        print(', '.join(histology_results.query('pattern == @i').sort_values('membership')['g'].tolist()))
    print("\n")


# In[ ]:




