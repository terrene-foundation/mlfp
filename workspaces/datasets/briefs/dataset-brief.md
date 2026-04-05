# Dataset Curation Brief

## Shared Drive Location

All datasets live on Google Drive shared folder `ascent_data` (folder ID: `16c3RkGmiwMWbjD7cJKbJx-JRZlgmQdws`), organized by module subfolder.

## Quality Criteria

Every dataset must be:
1. **Messy** — Missing values, mixed types, outliers, duplicates (real data is never clean)
2. **Sized for exercises** — 1,000-500,000 rows (meaningful ML, manageable in Colab)
3. **Publicly available** — Distributable under open license
4. **Singapore/APAC relevant** — Local context preferred
5. **Polars-compatible** — Loads cleanly with `pl.read_csv()` or `pl.read_parquet()`

## Module Datasets

### ascent01/ (Statistics, Probability & Data Fluency)
**Existing**: HDB prices, election data, happiness index, COVID panel, sentiments
**Action**: Merge HDB prices with MRT/school data from ascent_assessment for multi-table complexity
**Source**: data.gov.sg (HDB Resale Flat Prices), ascent_assessment parquets
**New needed**: Singapore weather station data (NEA), Singapore taxi trips (LTA, schema drift), e-commerce A/B test data

### ascent02/ (Feature Engineering & Experiment Design)
**Existing**: CPI, HDB prices, Toto history, dirty cafe sales
**Action**: Add Singapore economic indicators + healthcare ICU data + experiment data
**Source**: data.gov.sg, World Bank Open Data (APAC subset)
**New needed**: Healthcare ICU synthetic (MIMIC-style, 60K stays), e-commerce experiment data (500K users, SRM issues)

### ascent03/ (Supervised ML — Theory to Production)
**Existing**: Bank data, credit card, customer value, heart, housing, wine, zoo
**Action**: Replace with credit scoring + Lending Club. This is where supervised ML lives now.
**New needed**:
- Singapore credit scoring (synthetic, 100K apps, 12% default, protected attributes, leakage trap, 30% missing income)
- Lending Club loans (300K+, 150 features, real messiness)

### ascent04/ (Unsupervised ML, NLP & Deep Learning)
**Existing**: Netflix, online retail, Reddit news, text emotion
**Action**: Add e-commerce customer data, fraud data, news corpus, medical images
**New needed**:
- E-commerce behavioral dataset (200K txns, 50K customers, text reviews, Singapore market)
- Singapore news corpus (50K articles, CC-licensed, multi-topic, temporal)
- ChestX-ray14 subset (10K images, multi-label)
- Keep credit card fraud from Kaggle (284K txns, 0.17% fraud)

### ascent05/ (LLMs, AI Agents & RAG Systems)
**Existing**: COVID symptoms, credit card, customer value, housing, text emotion, trained models
**Action**: Reuse M3-M4 datasets (agents reason over familiar data). Add documentation corpora.
**New needed**:
- Kailash SDK documentation corpus (for RAG exercise)
- Singapore regulatory corpus (AI Verify, PDPA guidelines)

### ascent06/ and ascent06-dl/ (Alignment, Governance, RL & Deployment)
**Existing**: Emotions, Netflix, Reddit, RFM, spam, stock, wiki, MNIST, mask wearing, UCI-HAR, Shakespeare
**Action**: Keep text emotion for NLP baseline. Add alignment + governance datasets.
**New needed**:
- Domain Q&A pairs for SFT (1000 pairs, Kailash SDK domain)
- Preference pairs for DPO (500 pairs)
- Singapore parliamentary Hansard (CC-licensed)
- Custom inventory management Gymnasium environment

### ascent_assessment/ (Capstone)
**Existing**: 6 parquets (resi_sales, places, reviews, images, MRT, schools)
**Action**: Keep all. These are the integrated capstone dataset.
**Status**: Complete and ready.

## Data Creation Plan

For synthetic datasets (credit scoring, supply chain, e-commerce):
1. Define realistic schema with domain expert (instructor)
2. Generate using polars with controlled noise, imbalance, and missing patterns
3. Validate with DataExplorer (from kailash-ml)
4. Upload to shared Drive in parquet format
5. Document in `docs/dataset-descriptions.md`

## File Format Standards

- Parquet preferred (fast, typed, compact)
- CSV acceptable for simple tabular data
- JSON for nested/document data
- No Excel files (.xlsx) — convert to parquet or CSV
- Maximum 100MB per file, 500MB per module folder
