#!/bin/bash

URL=http://www.informatics.jax.org/downloads/reports/gene_association.mgi.gz
curl -O "$URL" && gunzip -f "${URL##*/}"
wait

URL=http://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt
curl -O "$URL" && gunzip -f "${URL##*/}"
wait

URL=http://aber-owl.net/media/ontologies/PhenomeNET/1/phenomenet.owl
curl -O "$URL" && gunzip -f "${URL##*/}"
wait

