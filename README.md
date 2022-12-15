# UniCell Deconvolve Paper
----------------------------------
This repository contains notebooks and scripts used in the UniCell Deconvolve paper, for the purposes of demonstrating how the figures, tables, and analyses presented in the paper were generated. Fully reproducing this analysis requires additional steps including downloading significant amounts of external data, in addition to changing filepaths. Some of these files may be updated in the near future.

For tutorials on using UniCellDeconvolve, please see the full documentation available at https://ucdeconvolve.readthedocs.io/en/latest/ and download the software package at https://github.com/dchary/ucdeconvolve/tree/main/ucdeconvolve

Interpretable & Context-Free Deconvolution of Multi-Scale Whole Transcriptomic Data With UniCell Deconvolve
----------------------------------
Authors: Daniel M. Charytonowicz, Rachel Brody, and Robert S. Sebra1

We introduce UniCell: Deconvolve Base (UCDBase), a pre-trained, interpretable, deep learning model to deconvolve cell type fractions and predict cell identity across Spatial, bulk-RNA-Seq, and scRNA-Seq datasets without contextualized reference data. UCD is trained on 10 million pseudo-mixtures from the world's largest fully-integrated scRNA-Seq training database comprising over 28 million annotated single cells spanning 840 unique cell types from 8989 studies. We show that our UCDBase and transfer learning models (UCDSelect) achievs comparable & superior performance on in-silico mixture deconvolution to existing, reference-based, state-of-the-art methods. Feature feature attribute analysis uncovers gene signatures associated with cell-type specific inflammatory-fibrotic responses in ischemic kidney injury, discerns cancer subtypes, and accurately deconvolves tumor microenvironments. UCD identifies pathologic changes in cell fractions among bulk-RNA-Seq data for several disease states. Applied to novel lung cancer scRNA-Seq data, UCD annotates and distinguishes normal from cancerous cells. Overall, UCD enhances transcriptomic data analysis, aiding in assessment of both cellular and spatial context.
