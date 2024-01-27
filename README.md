# Aviation-accident-visualization
GPT-4 Assisted Categorization and Visualization of NTSB UAV Accident Reports

## Introduction
This repository contains two Python scripts for the analysis and visualization of aviation accident data from the National Transportation Safety Board (NTSB). Utilizing Python and OpenAI's GPT-4, these scripts categorize and present insightful visualizations of the data.

### 1. NTSB_analysis_with_gpt4_V3.0.py

#### Purpose
Automates the categorization of NTSB aviation accident reports using OpenAI's GPT-4.

#### Description
The script processes National Transportation Safety Board (NTSB) aviation accident 
reports. It uses the OpenAI's GPT-4 API to categorize those reports based on the predefined 
categories outlined in the NTSB AVIATION OCCURRENCE CATEGORIES document.

#### Functionality
- Reads and processes CSV files with dynamic encoding detection.
- Sanitizes text data using unidecode, applied in parallel.
- Categorizes text using GPT-4, with caching to reduce redundant API calls.
- Efficient handling and filtering of DataFrame data.

#### Requirements
- Python 3.x
- Libraries: pandas, numpy, openai, joblib, chardet, unidecode
- OpenAI API key and CSV file paths set as environment variables.

#### Source Data
- NTSB Aviation Accident Reports: https://www.ntsb.gov/Pages/AviationQueryV2.aspx
- NTSB AVIATION OCCURRENCE CATEGORIES document https://www.ntsb.gov/safety/data/Documents/datafiles/OccurrenceCategoryDefinitions.pdf

#### Author Information
- Author: Eugene Pik
- Contact: LinkedIn https://www.linkedin.com/in/eugene/
- Date: January 5, 2024
- Version: 3.0

#### License
Licensed under the Apache License 2.0. For more details, visit https://www.apache.org/licenses/LICENSE-2.0.html.


### 2. NTSB_reports_visualisation_and_map_V1.1.py

#### Purpose
Analyzes UAV accident data from the NTSB, presenting it in various graphical formats and an interactive map.

#### Description
"Unmanned Aircraft NTSB Accident Probable Cause Visualization and Interactive Map Script" by Eugene Pik is focused on analyzing and visualizing unmanned aviation accident data from the National Transportation Safety Board (NTSB). This Python script employs various libraries to process, categorize, and present data in graphical forms and an interactive map, offering insights into accident trends, categories, and geographical distributions.

#### Functionality
1. Data Loading and Validation: Loads aviation accident data from CSV files and validates the presence of required columns. Implements robust error handling for data integrity.
2. Category Extraction and Mapping: Parses categories from accident data using regular expressions, mapping these to descriptive names.
3. Data Visualization:
   - Time Series Analysis: Plots accidents over time, analyzing seasonal trends.
   - Category Analysis: Generates bar and radar charts for visualizing accident categories and UAV model comparisons.
   - Geographical Mapping: Creates an interactive map with detailed markers, displaying accidents by location and category.
4. Additional Data Processing: Includes conversion of month numbers to names, retrieval of country codes, and formatting of accident details.
5. Custom Plot Functions: Functions for saving plots and setting common plot styles for consistent visualization.

#### Requirements
- Python 3.x
- Libraries: pandas, matplotlib, seaborn, folium

#### Source Data
- CSV file created by the script "NTSB_analysis_with_gpt4_V3.0.py"

#### Author Information
- Author: Eugene Pik
- Contact: LinkedIn https://www.linkedin.com/in/eugene/
- Date: January 27, 2024
- Version: 1.1

#### License
Licensed under the Apache License 2.0. For more details, visit https://www.apache.org/licenses/LICENSE-2.0.html.
