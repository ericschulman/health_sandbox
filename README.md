# Health Sandbox
This is an undocumented repository with statistical exercises exploring publicly available data from the Centers for Medicare & Medicaid Services (CMS). All data is presented as it was found on the CMS website for my own personal use.
* The exercises are exploratory in nature for my own personal use and involve joining the data and understanding variable names definitions, assessing the viability of model evaluation and machine learning techniques in the area of public health, and exploring the predictive fit of machine learning techniques with publicly available data. 
* It is important to note that causal conclusions cannot be drawn from these exercises, as the techniques being explored are not explicitly designed for causal inference and are known to have issues with making causal conclusions.
* Those interested in pre-merged files from CMS and exploring the predictive fit of machine learning techniques with publicly available data may find the code in this repository of interest.

## `aca_mkt`
This contains code related to the ACA public market place. Much of this code was written in conjunction with Himani Verma as part of the University of Texas's undergraduate research fellowship program. 
- `analysis`: exploration and regression analysis (boosted trees, logit, ols, iv) for 2016 and combined 2016-2017 data
- `analysis_2017`: regression analysis (boosted trees, logit) on 2017 data
- `data`: raw data files for 2016-2019, plans data, and processed data
- `pooled_predictions`: pooled 2016 and 2017 merged characteristics file with dummy variables for year
- `preprocess`: data pre-processing and merge of issuer and county characteristics 

## `medicare_adv`
Many of the same techniques/code applied in the `aca_mkt` folder have been applied to [publicly avaiable data](https://www.cms.gov/Medicare/Health-Plans/MedicareAdvtgSpecRateStats/DataFiles) on Medicare Advantage plans.

## `ESRD`
Public data on medicare beneficiaries with end stage renal disease. The excersizes relate to geocoding locations to better understand the geographic distribution of anonymized data.

