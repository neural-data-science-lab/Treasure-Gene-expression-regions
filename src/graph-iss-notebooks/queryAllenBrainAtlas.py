#!/usr/bin/env python
# coding: utf-8

# # Query Allen Brain Atlas
# Download expression grids from Allen Brain Atlas for each gene present in the taglist

# In[4]:


import requests
import re
import zipfile
import urllib
import array
import pandas as pd
import numpy as np
import os


# In[5]:


# Path where to save the expression grids
path = '../../data/graph_iss_data/graph-iss/'
ABA_path = path + "data/AllenBrainAtlas"

# Gene taglist
tagList = pd.read_csv(path + "data/tagList_99-gene.csv", sep = ",", usecols = [1], header = None, names = ["Seq"])
tagList = tagList.Seq.unique()

# Download energy grids
for gene in tagList:
    print(str(gene))
    if not os.path.exists(ABA_path+"/"+gene):
        os.makedirs(ABA_path+"/"+gene)
    query_string = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model::SectionDataSet,rma::criteria,[failed$eq'false'],products[abbreviation$eq'Mouse'],plane_of_section[name$eq'coronal'],genes[acronym$eq'"+gene+"']"
    response = requests.get(query_string)
    res=re.findall('<id>(.*)</id>',response.text)
    for exp in res:
        print("\t"+str(exp))
        if not os.path.exists(ABA_path+"/"+gene+"/"+exp):
            os.makedirs(ABA_path+"/"+gene+"/"+exp)
        query_string = "http://api.brain-map.org/grid_data/download/"+exp+"?include=energy"
        fh = urllib.request.urlretrieve(query_string)
        zf = zipfile.ZipFile(fh[0])
        header = zf.read('energy.mhd')
        raw = zf.read('energy.raw')
        arr_size=np.array(re.split(" ",re.findall('DimSize = (.*)\nElementType',header.decode("utf-8"))[0])).astype(int)
        arr = np.array(array.array('f',raw)).reshape(arr_size, order='F')
        np.save(ABA_path+"/"+gene+"/"+exp+"/energy",arr)

print(f'{ABA_path = }')
