"""
NTSB Aviation Accident Probable Cause Analysis and Categorization Script by Eugene Pik. 
The script processes National Transportation Safety Board (NTSB) aviation accident 
reports sourced from https://www.ntsb.gov/Pages/AviationQueryV2.aspx. 
It utilizes the OpenAI's GPT-4 API to categorize reports based on the predefined 
categories outlined in the NTSB AVIATION OCCURRENCE CATEGORIES document 
https://www.ntsb.gov/safety/data/Documents/datafiles/OccurrenceCategoryDefinitions.pdf.


Functionality:
- Reads and processes CSV files with dynamic encoding detection.
- Sanitizes text data using unidecode, applied in parallel.
- Categorizes text using OpenAI's GPT-4, with caching to reduce redundant API calls.
- Efficient handling and filtering of DataFrame data.

Requirements:
- Python 3.x
- Libraries: pandas, numpy, openai, joblib, chardet, unidecode
- OpenAI API key and CSV file paths set as environment variables.

Usage:
- Ensure that the required libraries are installed with the correct versions.
- Set environment variables for the OpenAI API key and file paths.
- Run the script.

Author: Eugene Pik
Contact: LinkedIn https://www.linkedin.com/in/eugene/
Date: January 5, 2024
Version: 3.0
License: This script is free for use and distribution under the terms of the Apache License 2.0.
For more details, see: https://www.apache.org/licenses/LICENSE-2.0.html
"""

import re
import os
import time
import chardet
import logging
import numpy as np
import pandas as pd
from openai import OpenAI
from unidecode import unidecode
from joblib import Parallel, delayed

# Initialize logging for error reporting and debugging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
client = OpenAI()

# Load API key and file paths from environment variables
client.api_key = os.getenv("OPENAI_API_KEY")
if not client.api_key:
    logging.error("OpenAI API key is not set. Please check your environment variables.")
    exit(1)

df_path = os.getenv("NTSB_UNMANNED_CSV_PATH", 'c:/NTSB/NTSB_unmanned.csv')
categories_path = os.getenv("NTSB_CATEGORIES_CSV_PATH", 'c:/NTSB/NTSB_categories.csv')
output_path = os.getenv("NTSB_OUTPUT_CSV_PATH", 'c:/NTSB/NTSB_unmanned_with_categories_gpt4_V3.csv')

def detect_encoding(file_path, sample_size=10000):
    """
    Detects file encoding by reading a sample of the file.
    
    :param file_path: Path to the file.
    :param sample_size: Number of bytes to read for the sample.
    :return: Detected encoding.
    """
    with open(file_path, 'rb') as file:
        sample = file.read(sample_size)
        result = chardet.detect(sample)
        return result.get('encoding', 'utf-8') or 'utf-8'

def read_csv_utf8(file_path):
    """
    Reads a CSV file with UTF-8 or detected encoding and returns a DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame after reading the CSV.
    """    
    try:
        encoding = detect_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding, low_memory=False)
    except Exception as e:
        logging.error(f"Failed to read file {file_path}. Error: {e}")
        raise

def apply_unidecode_to_chunk(df_chunk):
    """
    Applies unidecode to each string element in a DataFrame chunk for text normalization.
    
    :param df_chunk: DataFrame chunk.
    :return: Processed DataFrame chunk.
    """
    return df_chunk.applymap(lambda x: unidecode(x) if isinstance(x, str) else x)

def parallel_unidecode(df, n_jobs=-1):
    """
    Applies unidecode to a DataFrame in parallel to enhance performance.
    
    :param df: DataFrame to process.
    :param n_jobs: Number of jobs to run in parallel.
    :return: DataFrame after processing.
    """    
    if df.empty:
        return df
    n_jobs = max(1, os.cpu_count() if n_jobs == -1 else n_jobs)
    df_chunks = np.array_split(df, n_jobs)
    return pd.concat(Parallel(n_jobs=n_jobs)(delayed(apply_unidecode_to_chunk)(chunk) for chunk in df_chunks))

def load_and_format_categories(file_path):
    """
    Loads and formats the categories from a CSV file.
    :param file_path: Path to the categories CSV file.
    :return: List of formatted categories.
    """
    categories_df = read_csv_utf8(file_path)
    formatted_categories = []
    for _, row in categories_df.iterrows():
        # Check if CategoryDescription is a string, otherwise handle NaN
        if isinstance(row['CategoryDescription'], str):
            category_description = row['CategoryDescription'].replace('"', '\\"')
        else:
            category_description = row['CategoryDescription']  # Keeps NaN as is

        formatted_category = (
            f"{{"
            f"\"Category Name\": \"{row['CategoryName']}\", "
            f"\"Category Code\": \"{row['CategoryCode']}\", "
            f"\"Category Description\": \"{category_description}\""
            f"}}"
        )
        formatted_categories.append(formatted_category)
    return formatted_categories

# Global cache dictionary
cache = {}

def categorize_probable_cause(text, model="gpt-4-0613"):
    """
    Categorizes the probable cause of an aviation accident using OpenAI's GPT-4.
    :param text: Text describing the probable cause.
    :param model: GPT-4 model to be used for categorization.
    :return: Categorization result.
    """    
    # Check cache first
    if text in cache:
        logging.info("Cache hit for text")
        return cache[text]

    try:
        categorization_instruction = (
            "Analyze Category Name and Category Description of each category. Assign the top 3 most suitable Category Codes "
            "of different categories from that list, along with their confidence percentages to the user provided text. "
            "Format your response as [{CategoryCode1:confidence1},{CategoryCode2:confidence2},{CategoryCode3:confidence3}] "
            "without any additional wording. "
            "Response example: [{LOCI:70},{CFIT:60},{NAV:50}] "
            "Use the following list of categories: ".join(categories)
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": categorization_instruction},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=150  # Increase max_tokens if necessary
        )

        # Parse the response to extract categories and confidence percentages
        parsed_response = parse_response(response.choices[0].message.content.strip())
        
        # For debugging purposes, print the parsed response to see what was returned from OpenAI.
        # print("Parsed Response:", parsed_response)
        # print("Raw Response from OpenAI:", response.choices[0].message.content.strip())


        # Store the parsed response in cache
        cache[text] = parsed_response
        return parsed_response
    except Exception as e:
        logging.error(f"Error in categorizing Probable Cause for text '{text}': {e}")
        return "Uncategorized"

def parse_response(response_text):
    """
    Parses the response text to extract category codes and their confidence percentages.
    
    :param response_text: Text response from GPT-4.
    :return: List of categories and confidence percentages.
    """    
    pattern = r'\{"?([A-Z]+)"?:\s*(\d+)\}'
    matches = re.findall(pattern, response_text)

    # Initialize with default values if not enough matches
    categories_with_confidence = [("Uncategorized", 0), ("Uncategorized", 0), ("Uncategorized", 0)]

    # Update with actual values from matches
    for i, match in enumerate(matches[:3]):  # Limit to top 3 matches
        categories_with_confidence[i] = (match[0], int(match[1]))

    return categories_with_confidence


if __name__ == "__main__":
    try:
        # Load and process CSV files
        df = read_csv_utf8(df_path)
        if df.empty:
            logging.info("DataFrame is empty. Exiting script.")
            exit(0)

        categories = load_and_format_categories(categories_path)
 
        # Apply text sanitization in parallel
        df = parallel_unidecode(df)
        
        # Data cleaning and preparation steps
        df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')
        df = df.dropna(subset=['EventDate', 'ProbableCause', 'Latitude', 'Longitude'])
        df = df[(df['ReportStatus'] == 'Completed') & 
                (df['ProbableCause'].notna() & df['ProbableCause'].str.strip().astype(bool)) &
                (df['Latitude'].notna() & df['Longitude'].notna())]
        df = df.drop(['DocketUrl', 'DocketPublishDate'], axis=1)
        df['Year'] = df['EventDate'].dt.year
        df['Month'] = df['EventDate'].dt.month


        # Apply categorization and split into new columns
        for index, row in df.iterrows():
            category_data = categorize_probable_cause(row['ProbableCause'] if pd.notnull(row['ProbableCause']) else "Uncategorized")
            for i in range(3):
                df.at[index, f'CategoryCode_{i+1}'] = category_data[i][0]
                df.at[index, f'confidence_{i+1}'] = category_data[i][1]

        # Save the processed DataFrame with new columns
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info("NTSB analysis completed.")
  
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        raise  # Re-raise the exception after logging    
    
