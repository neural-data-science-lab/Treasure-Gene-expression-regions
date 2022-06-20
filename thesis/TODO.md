## to include
- why do we analyze mice instead of humans?
- what is the actual question to answer?

- Dataset:
  - Allen mouse Brain Atlas
    - gene expression values:
      - non-pathogenic mice 
      - male/female?
    - how is closeness measured?
      - structural ontology
        - siblings in hierarchy 
        |-> predict from parents?
          - how are gene expressions in superstructures aggregated? DeepGO similar stuff
      - define baseline method for that purpose 
    
        
        


1. analysis of graph neural networks for gene expression patterns
  - Hypothesis:
    - is gene expression predictable in from close tissues?
      - gene_expr correlation graph with clearly visible vertical lines
  - model analysis
    - predictive performance even worse with flat model
    - motivate choice of different features for this purpose 
      - molecular features/genetic footprint
      - phenotypical features/functional footprint -> copy some stuff from DTI-Voodoo
      - gene expression values for fix-point learner 
    - show applicability of background knowledge such as PPI graphs
      - why would the model benefit here? Show intuition
    - show applicability through literature -> Roberts comments and literature links -> copy and explain some main findings 

2. embeddings analysis with UMAP and alterations of it
  - UMAP with ontology graph as underlying graph
