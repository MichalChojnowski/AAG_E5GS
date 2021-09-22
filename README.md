# AAG-Vantage

This repository is created to host scripts, notebooks, and configurations needed to support a **Vantage Site Selection** case.  While the specific analyses can depend case-to-case, this repository hosts a variety of scripts to guide model development and the core components of a site selection project.

To view a sample of the **Vantage demo** created for both Retail and Restaurant clients, please see the following [link](https://bain-vantage-dev.carto.solutions/login).  Please contact Bilal Lodhi (bilal.lodhi@bain.com) or Brian Tunnell (brian.tunnell@bain.com) for access credentials and with any questions.

The broader **home for AAG location intelligence** can be found at the following [confluence page](https://bainco.atlassian.net/wiki/spaces/AAG/pages/14830963651/Location+Intelligence).  This page is home to many resources and guides related to all things location intelligence at Bain.  

Further, there are a variety of **fantastic tutorials and common LI use-cases** put together by teammates within the EMEA & US PEG ringfences to assist with whitespace cases.  This home for general location intelligence tools and techniques can be accessed in this [repository](https://github.com/Bain/aag-location-intelligence).

As part of Vantage software delivery, we partnered with **Carto** to develop and deliver a leave-behind site selection application with clients.  The repository hosting the Vantage application can be found [here](https://github.com/CartoDB/bain-vantage).  This repository hosts the core application and both front-end and back-end components.  Reach out to Bilal Lodhi, Brian Tunnell,  Rafa Vaquero (rvaquero@carto.com), or your case's Carto project manager for access.  The bain-vantage repository hosts the broader Vantage application and includes a sub-directory (`api/analysis/algorithms`) which contains sample notebooks for each step of the whitespace process and can be updated with code developed during each case.  These notebooks are also included within this repository ([carto_tutorials](https://github.com/Bain/aag-vantage/tree/master/src/notebooks/carto_templates)), and serve as the foundation for development of the key steps of the whitespace process and integration within the Vantage application.

Carto help guides and references included here:
1. https://carto.com/help/working-with-data/
2. https://carto.com/developers/data-services-api/reference/#introduction

## Getting Started
To get started:
1. Clone the aag-vantage repo

    ` $ git clone git@github.com:Bain/aag-vantage.git`
    
2. Download the conda environment file with requisite dependencies

    `$ conda env create --file vantage_env.yml`
    
    `$ conda activate vantage`
    
3. Create a branch for local development:

    `$ git checkout -b your_branch_name master`

4. Reach out to Bilal Lodhi to set up a Carto account and credentials if not already obtained or not using bainadmin account

5. Request access from Rafa Vaquero (rvaquero@carto.com) to the bain-vantage repository to fascilitate development and integration of code within Vantage application

6. Update sample_credentials.ini with relevant carto account credentials used to store data and develop code

Contributions are welcome and appreciated!  As we continue to work on site selection cases, we hope to build out our library of functions and use-cases.

## Overview

The goal of this repository is to host building blocks for future site-selection projects and Vantage application delivery.  Currently, the work is centered around Site Selection and Whitespace analysis use cases, but future areas of development include:
- Footprint Rationalization
- Footprint Expansion
- Network Optimization
- In-store mapping/Layout

At it's core, the whitespace analysis process revolves around training a model to predict store performance using external factors and using that model to screen a territory for attractive locations.  For more detailed information regarding the Whitespace process, please see the Technical How-to-Guide in addition to other related materials in the project case cloud.

The key steps to cover during a site selection Vantage case include:
1. Align on modeling approach and priorities
    - What success metric does the client care about?  Is it annual revenue?  Revenue/sq ft?  Number of customers?
    - How does the client currently think about expansion, cannibalization, and site evaluation?
    - How does the existing site selection process work?
    - Who are the main stakeholders involved (e.g. franchisee system, or all corporate-owned)?
    - What is the "catchment" area around each store (e.g. 1 mile radius, 5 minute drive-time isochrone)?
    - What locations should be used as seeds for the Whitespace analysis?
2. Store data ETL
    - Full template data request located in case cloud, but typical columns include
        - Active stores (1 row per store, with location information and store id)
        - Historical performance data (Ideally 2+ years, but at least last twelve months or previous fiscal year)
            - If creating historical cannibalization model, see if client has already aggregated impact database or request longer time-span of performance data to create dependent variable
        - Stores in development (if applicable)
            - Future pipeline locations can be used in tool and as seed locations
    - Work with Carto team to align on key tables needed for Vantage app and requisite ETL process
    - Typical performance model target variables include Yearly Sales, Sales/SqFt, or # of Patients/Customers
    - If creating cannibalization model, need access to historical sales and opening and close data to create historic cannibalization event database and target variable
        - In previous cases, have used target of year over year change in sales relative to control population (unimpacted stores within the same DMA)
3. Enrich client locations
    - In order to generate a whitespace model, we need to "enrich" our existing locations with relevant external data sources
    - If client does not have existing data subscriptions, recommend to use sources provided through the Carto Data Observatory to enrich locations due to ease of integration.  Further information available in the Data Mapping file in the case cloud.
    - Trade Area size and type can affect model performance (e.g. using 5 minute drive-time or 3 mile radius).  Trade Area/Catchment Area approaches typically involve either straight-line radii, or travel distance or time based "isochrones" (e.g. 3 minute drive-time trade area).  Typical approaches include:
        - Using a fixed radius or isochrone for each store
        - Varying radius/time by a category (e.g. store format or urbanicity)
        - Incorporating multiple trade area types and sizes (almost like a model parameter) to test in model (e.g. POIs within 1 mile, Population within 3 miles, etc.) and prioritize most predictive
        - Utilizing human mobility patterns to create "intelligent" trade areas (requires heavier iteration and effort to incorporate in whitespace model)
    - For cannibalization model, have crafted features around the distance, population, shared area, and rank-order of the locations (existing store and new store locations)
    - Any enrichments performed at this stage will need to be scalable, as they will need to be performed for your seed location database as well (potentially upwards of 100K seed locations depending on granularity chosen)
    - Important to utilize defined helper functions (if changes made, align with Carto backend team) to enrich locations and help with ease of integration for Drop-A-Pin model
    - Can adjust enrichment datasets, variables, and aggregations in `enrichment_vars.json`
4. Driver analysis and predictive modeling
    - Examine relationship between target variable and key drivers of performance
    - Preprocess & clean any data for prediction as needed
    - Train and test different model types, iterate with case team on features crafted
    - Examine model accuracy across regions and geographies
    - Decide if using cannibalization model or heuristic
    - Depending on type of client, may require multiple models per format or region (e.g. separate model for free-standing drive through stores vs. mall stores)
5. Model tuning & Selection
    - Tune model performance by iterating on features and model parameters
    - Model hyperparameter tuning set up using hyperopt and mlflow
    - Save model & pre-processing pipeline to use to score seed locations and to share with Carto to integrate for Drop-a-Pin predictions for the tool
6. Enrich & score seed locations
    - Align with client on preferred style of seed location:
        - Administrative Region Centroids
            -Pros: Relatively consistent, coincide with many aggregation methodologies, easily explainable
            -Cons: Lack actionability dependent on client desired granularity
        - Grid centroids
            -Pros: Consistent, can use to create whitespace clusters or heat maps
            -Cons: Can be computationally heavy to enrich and process at scale, may require additional pre-filtering based on heuristics (e.g. must have at least X population)
        - POIs (e.g. competitors, traffic generators, clusters of retail activity, shopping centers)
            -Pros: Actionable as it based off of existing real estate locations
            -Cons: Requires base data and methodology alignment with client, may miss true Greenfield opportunities
    - Using features selected for model, enrich seed locations using same process
        - Utilize Carto helper functions and notebooks located in Vantage repo to assist with enrichment
        - Enrichment code will need to be integrated for Drop-A-Pin predictions within the application
    - Predict using developed model(s) to score seed locations predicted target variable and assist with prioritization
7. Prioritize seed locations
    - Create master whitespace database with:
        1. Scored seed locations with target variable
        2. Any pertinent post-processing variables to assist with additional calculations (such as urbanicity & region for profit margin and cost of development assumptions to enable cash on cash calculation)
        3. Any filtering columns (e.g. only use seeds with X population density)
    - If developing cannibalization model, provide scored Seed x Site predicted cannibalization estimates to enable filtering (e.g. only allow cannibalization estimate of 10%)
    - Typical approach to prioritize seed locations is:
        1. Filter based on heuristics informed by existing footprint(e.g. must have X population density)
        2. Filter based on target variable(s) (e.g. Predicted Revene >= $Y)
        3. Filter based on existing client footprint:
            - No-go zones by category (e.g. 2 miles for urban stores, 3.5 miles for suburban, etc.)
            - Max % overlap (e.g. using catchment areas for each store, only allow % overlap <= X%)
            - Predicted cannibalization (e.g. Predicted Max Cannibalization <= Z%)
        4. Filter to prioritize seed locations:
            - Same as above, but with potential seed x seed cannibalization
            - Prioritizing seed locations to avoid recommending too many locations in one area
            - Can also cluster results into final recommendation
    - Recommend to test multiple assumptions with case team, as well as to compare results with existing footprint & competitive benchmarks 
8. Integrate & visualize results
    - Need to provide Carto with multpile key inputs to use in the Vantage tool:
        1. Datasets: Any datasets used in the modeling, scoring, or filtering process
        2. Enrichment: Trade areas, features created, and variables aggregated for modeling or filtering (e.g population within y miles)
        3. Preprocessing: Any data pre-processing steps (e.g. StandardScaler, LabelEncoder, etc.) used before model prediction
        4. Models: Predictive model(s) used to enable drop-a-pin predictions
        5. Post-processing: Any post-prediction calculations or assumptions (e.g. profitability assumptions by urbanicity and DMA)
        6. Whitespace Tables: Master output table with scored seed locations and any tables related to cannibalization (e.g. Predicted Seed x Site cannibalization)
        7. Whitespace Filtering: Algorithms, input parameters & steps used to filter and prioritize seed locations
    - Carto table & process documentation is provided in the bain-vantage repository & in the set-up power-point
    - Work with Carto PM & backend engineer to align on any process changes and tool implications
    
## Features

The repository is organized to contain helper functions and sample notebooks related to data enrichment, driver analysis, and whitespace analysis.  

Within the Vantage tool, clients will have the ability to **"drop-a-pin"** and perform a prediction for any point in the map.  This means that all geospatial operations, feature engineering, pre-processing, and prediction pipelines will need to be integrated within the Vantage Carto environment in order to enable live predictions.  To assist with this, we have co-created core geospatial functions and Carto I/O operations within the repository [cf_model.py](https://github.com/Bain/aag-vantage/blob/master/src/enrichment/enrichment.py).  It is important to try and leverage these helper functions when possible, as they have been co-created with the Carto engineering team and will fascilitate ease of integration within the broader application.  When updating methods, align with Carto team on core changes needed to ensure clarity and visibility on tool requirements.

Another key feature of the tool is the ability to run a **whitespace analysis** based on pertinent input parameters.  It is important to provide Carto with a few peices to enable this analysis:
1. Whitespace Master table with scored seed locations
2. Filtering parameters used to filter locations out
3. Prioritization algorithm used to hone in on top sites

### Core repository materials are hosted in the src and organized as follows:
- constants: 
    - global_constants.py: core constants and assumptions used during the modeling or whitespace process (e.g. meters per mile)
- enrichment: 
    - enrichment.py: carto class containing core enrichment and carto I/O functions
    - enrichmet_dbs.json: example enrichment template file to assist with external dataset enrichment.  Key parameters include:
        1. `DATASET_ID`: Carto dataset id to use for enrichment
        2. `VAR_IDS`: Variable ids to pull in and use for model development
        3. `VARS_TO_CONSOLIDATE`: Higher-level variables to aggregate from more granular raw data (e.g. roll up population by age into higher-buckets)
        4. `PERC_COLS_TO_CALC`: Columns to standardize by providing the numerator and denominator (e.g. transform population by age into % of total population by age)
- models:
    - models.py: WhitespaceModels class with helper functions designed to enable model training, tracking using MLflow, and parameter tuning utilizing hyperopt
- notebooks:
    - carto_templates: Notebooks co-created with Carto to enable key steps of the Vantage process.  These are specifically centered around aspects that are integrated within the Vantage tool (drop-a-pin enrichment and prediction, and whitespace master table generation).  It is important to modify code to be consistent with this structure and format to ensure ease of integration within Vantage tool
        - 1_Generate_ws_seed_locations.ipynb: Create seed locations using commercial center clusters
        - 2_3_Enrich_predictions_seeds.ipynb: Enrich seed locations using drivers identified during modeling phase, predict using model(s) generated on existing store performance
        - 4_Get_cannibalization_events.ipynb: Generate a list of cannibalization events given existing stores, seed locations, and criteria to consider for cannibalization (e.g. 10 mile buffer around each store)
        - 5_Enrich_cannibalization_events.ipynb: Enrich cannibalization events with drivers identified during modeling phase, if no cannibalization model can skip this step
        - 6_predict_cannibalizations.ipynb: Predict cannibalization events using features and model trained on historical cannibalization events.  If no cannibalization model can skip this step.
    - Cannibalization Model.ipynb: Notebook created to demonstrate example cannibalization model and process.  Shows options for model approach and heuristic approach.  Requires "Cannibalization Event Database" based on historical site openings and observed changes in sales pre-post opening.
    - Driver Analysis.ipynb: Notebook overviewing store performance driver analysis.  Not exhaustive, but is set up to facilitate model tracking using MLFlow and model tuning using Hyperopt.
    - Enrichment.ipynb: Notebook demonstrating enrichment process and functions for existing client footprint and stores to utilize in Driver Analysis.ipynb
    - Whitespae Analysis.ipynb: Example whitespace analysis using either heuristics or predicted cannibalization

## Typical Workflow
Typical steps in the workflow consist of:
### Getting started
### Data ETL
### Client enrichment, driver analysis, & model generation
### Seed location enrichment and prediction
### Outputs and Vantage integration
        
## Auto-Documentation
Documentation can also automatically be generated from classes and methods using docstrings in the codebase where available. Inside the docker run

```
cd docs
sphinx-apidoc -o source/ ../src/
make html
```

The sphinx-apidoc command will index the current existing code in the `../src/` path and the `make html` will generate 
human readable web output.   

The main file is `docs/build/html/index.html` so open this in a browser to see the results. 

The Sphinx tool can only show useful documentation automatically if you include structured comments while you program in the numpy friendly format demonstrated in the examples.  

It is good form to include the input and outputs type hint information as well so that method signatures are clear from the documentation alone.

For more info on best practices [see the data product playbook](https://github.com/Bain/playbooks/blob/master/data-analytics-product-playbook.md#methodologies-for-building-apps)
