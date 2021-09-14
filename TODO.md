## TODO:
1. gene expression per structure
- prepare DeepGOPlus features for present nodes in PPI graph
- build pytorch graph structure with node features
- build network
- run and enjoy

2. find gene-expression/pathology data

## Nico 12.05.:
- datasets
- MPG resources, how to access?
  - VPN?!
  - see email
- gene expression split
- ingo and Gutachter

- repo in group or private

## Nico 23.06.:
- take patch for each structure -> image
- plug into graph 
- run classification

## Robert 05.07.:

- normalize intensity over gene expr -> min/max and regression
three different normalization techniques (Sara DeepMOCCA), normalize over 
1. row
2. column
3. matrix
with varying biological interpretation

1. take region as input and predict gene expression
2. predict structure from gene expression pattern
3. predict structure from gene expression and images

image magick to convert images

- predict region from image 
- predict gene expression from function
- combine both 

- pathologic image of cancer -> predict cancer type from morphology
- simulate loss of function/expression by removing one node of graph

## Nico 24.08.:
- impact of PPI network
- predict gene expression with knowledge about closeness and without
-> fixed predictor

