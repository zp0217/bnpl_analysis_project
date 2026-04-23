# BNPL Regulatory Gap Analysis

Website: https://zp0217.github.io/bnpl_analysis_project/

## Project

**Buy Now Pay Later (BNPL) Regulatory Gap: Evidence for Consumer Protection Reform from CFPB Complaints**


## About

A data-driven analysis of the regulatory gap surrounding Buy Now, Pay Later services in
the United States, combining exploratory data analysis of 13,396 CFPB consumer complaints,
natural language processing of 4,791 consumer narratives (TF-IDF and LDA topic modeling),
and a structured regulatory gap analysis comparing BNPL disclosure practices with TILA
requirements for credit card issuers.

## Structure

```
bnpl_analysis/
├── _quarto.yml            # Site configuration
├── _publish.yml           # Publish target (gh-pages)
├── index.qmd              # Homepage
├── introduction/          # Introduction + literature review
├── data_source/           # Data sources and scope
├── EDA/                   # Exploratory data analysis
├── nlp/                   # TF-IDF, regulatory terms, LDA topic modeling
├── regulatory_gap/        # TILA vs. BNPL comparison
├── policy/                # Policy recommendations
├── conclusions/           # Conclusions and limitations
├── assets/                # references.bib, nature.csl(reference for paper) 
├── image/                 # Figures (PNG)
├── data/                  # Raw and processed data
└── docs/                  # Rendered site (GitHub Pages serves from here)
```


