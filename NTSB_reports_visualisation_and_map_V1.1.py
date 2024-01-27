"""
Unmanned Aircraft NTSB Accident Probable Cause Visualization and Interactive Map Script by Eugene Pik.

This script is designed to analyze and visualize unmanned aviation accident data from the National Transportation Safety Board (NTSB). Utilizing various Python libraries, it processes and categorizes data, presenting it in graphical formats and an interactive map. The script provides insights into trends, categories, and geographical distributions of aviation accidents.

# Functionality
1. Data Loading and Validation: Loads aviation accident data from CSV files and validates the presence of required columns. Implements robust error handling for data integrity.
2. Category Extraction and Mapping: Parses categories from accident data using regular expressions, mapping these to descriptive names.
3. Data Visualization:
   - Time Series Analysis: Plots accidents over time using line plots and analyzes seasonal trends.
   - Category Analysis: Generates bar and radar charts for visualizing accident categories and UAV model comparisons.
   - Geographical Mapping: Creates an interactive map with detailed markers, displaying accidents by location and category.
4. Additional Data Processing: Includes conversion of month numbers to names, retrieval of country codes, and formatting of accident details.
5. Custom Plot Functions: Functions for saving plots and setting common plot styles enhance visualization consistency.

# Requirements
- Python Version: Python 3.x.
- Libraries: Required Python libraries include `re`, `os`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `folium`, `pycountry`, and `statsmodels`.
- Data Files: CSV files with NTSB aviation accident data and category information.

# Usage
1. Setup: Ensure Python 3.x and all libraries are installed. Verify the availability and format of the CSV files.
2. Running the Script: Execute the script to process the data and generate visualizations.
3. Output: View output as line plots, bar graphs, radar charts, and an interactive map, providing comprehensive insights into the data.

Author: Eugene Pik
Contact: LinkedIn https://www.linkedin.com/in/eugene/
Date: January 9, 2024
Version: 1.1
License: Apache License 2.0. See: https://www.apache.org/licenses/LICENSE-2.0.html
"""

# Import necessary libraries
import re  # Regular expressions
import os  # Operating system interfaces
import sys
import logging
import calendar
import numpy as np  # Numerical Python for array operations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting library
import matplotlib.patches as mpatches
import seaborn as sns  # Statistical data visualization
import folium  # Interactive map creation
import pycountry  # Country codes and names
from math import pi
from itertools import cycle  # For cycling through iterables
from branca.element import Figure  # For advanced figure elements in maps
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# set path to the input CSV files - either environment variables or locations on drive C:
df_path = os.getenv("NTSB_UNMANNED_WITH_CATEGORIES_CSV_PATH", 'c:/NTSB/input/NTSB_unmanned_with_categories_gpt4_V3.csv')
categories_path = os.getenv("NTSB_CATEGORIES_CSV_PATH", 'c:/NTSB/input/NTSB_categories.csv')

def load_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            df = pd.read_csv(file)
            logging.info(f"Successfully loaded file: {file_path}")
            return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        print(f"Error: File not found - {file_path}. Exiting script.")
        sys.exit(1)  # Exits the script with a non-zero value to indicate an error
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error in file: {file_path} - {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading file: {file_path} - {e}")
        sys.exit(1)

# Validate if required columns exist in the DataFrame
def validate_data(df, required_columns):
    if not all(column in df.columns for column in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        logging.warning(f"Missing columns in DataFrame: {missing_columns}")
        return False
    return True


# Load data from CSV files
df = load_csv(df_path)
categories_df = load_csv(categories_path)

# Convert 'Year' and 'Month' to datetime format
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')

# Validate data
if not validate_data(df, ['Categories_and_confidence', 'Year', 'Month', 'Latitude', 'Longitude']):
    logging.error("Data validation failed. Required columns are missing.")
    sys.exit("Error: Data validation failed.")

if not validate_data(categories_df, ['CategoryCode', 'CategoryName']):
    logging.error("Categories data validation failed. Required columns are missing.")
    sys.exit("Error: Categories data validation failed.")

# Function to parse categories
def extract_categories(cell_value):
    """
    Extracts categories from a given cell value in the DataFrame using regular expressions.
    
    :param cell_value: Value of the cell from which categories need to be extracted.
    :return: A list of extracted categories, or an empty list if the cell value is NaN.
    """    
    if pd.isna(cell_value):
        return []
    return re.findall(r'\{([^:]+):', cell_value)

# Vectorized extraction of the first category
df['First_Category'] = df['Categories_and_confidence'].str.extract(r'\{([^:]+):')[0].fillna('Unknown')

category_mapping = dict(zip(categories_df['CategoryCode'], categories_df['CategoryName']))
df['CategoryName'] = df['First_Category'].map(category_mapping)

def save_plot(filename, folder_path='c:/NTSB/output', dpi=400):
    """
    Saves the current matplotlib plot as a PNG file with increased resolution.

    :param filename: Name of the file to save the plot.
    :param folder_path: The directory path where the file will be saved.
    :param dpi: Resolution of the output image in dots per inch.
    :return: None. The plot is saved to the specified path.
    """
    full_path = os.path.join(folder_path, filename)
    try:
        plt.savefig(full_path, dpi=dpi)
        logging.info(f"Plot successfully saved as: {full_path}")
    except Exception as e:
        logging.error(f"Failed to save '{full_path}': {e}")

def set_common_plot_style(title, xlabel, ylabel):
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.set_style("whitegrid")

# Function to format axis labels as integers
def format_axis_as_integers(ax, axis='both'):
    """
    Formats the axis labels of a plot as integers for enhanced readability.

    :param ax: The axes object of the matplotlib plot.
    :param axis: Specifies which axis ('x', 'y', or 'both') to format.
    :return: None. The function modifies the axes object in place.
    """    
    if axis in ['both', 'x']:
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    if axis in ['both', 'y']:
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))

# Function to plot number of accidents per year as a stacked bar graph
def plot_accidents_per_year(df):
    """
    Plots the number of unmanned aviation accidents per year as a stacked bar chart,
    with each stack representing a category.

    :param df: DataFrame containing unmanned aviation accident data.
    :return: None. Displays and saves the stacked bar plot of accidents per year.
    """
    # Use Seaborn style for better aesthetics
    sns.set_style("whitegrid")

    # Aggregate data by year and category
    yearly_category_counts = df.groupby(['Year', 'First_Category']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot stacked bar chart using the category colors
    yearly_category_counts.plot(kind='bar', stacked=True, 
                                color=[category_color_map.get(cat) for cat in yearly_category_counts.columns], 
                                edgecolor='black', linewidth=0.5, 
                                ax=ax, width=0.85)  # Adjusted bar width to 0.8

    set_common_plot_style('Number of Accidents per Year by Category', 'Year', 'Number of Accidents')

    # Customizing font size for the axis labels
    ax.set_xlabel('Year', fontsize=16)  # Increase font size for 'Year'
    ax.set_ylabel('Number of Accidents', fontsize=16)  # Increase font size for 'Number of Accidents'

    # Create a custom legend and adjust its position and font size
    legend_patches = [mpatches.Patch(color=color, label=cat) for cat, color in category_color_map.items()]
    # Adjusting legend position
    legend = plt.legend(handles=legend_patches, title='Categories', fontsize=14, loc='upper right', bbox_to_anchor=(1.008, 0.98)) # x and y location of the legend

    # Increase the font size of the legend title
    plt.setp(legend.get_title(), fontsize=16)

    # Adjust y-axis limit to allow space for the legend
    y_max = yearly_category_counts.sum(axis=1).max()
    ax.set_ylim(0, y_max * 1.1)  # Setting the upper limit to 110% of the maximum value

    # Print year numbers horizontally
    ax.set_xticklabels(yearly_category_counts.index, rotation=0, fontsize=14)

    # Instead of tight_layout, manually adjust the plot margins to ensure all elements fit
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    
    save_plot('plot_accidents_per_year.png')
    plt.show()

# Function to plot number of accidents per month as a stacked bar graph
def plot_accidents_per_month(df):
    """
    Plots the number of unmanned aviation accidents per month as a stacked bar chart,
    with each stack representing a category.

    :param df: DataFrame containing unmanned aviation accident data.
    :return: None. Displays the stacked bar plot of accidents per month.
    """
    # Use Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    # Aggregate data by month and category
    monthly_category_counts = df.groupby(['Month', 'First_Category']).size().unstack(fill_value=0)

    # Adjusting the figure size
    fig, ax = plt.subplots(figsize=(18, 7))  # Increased from (12, 7)

    # Plot stacked bar chart using the category colors, adding a black edge
    # Making bars 20% wider and adding a thin black border around categories
    monthly_category_counts.plot(kind='bar', stacked=True, 
                                 color=[category_color_map.get(cat) for cat in monthly_category_counts.columns], 
                                 edgecolor='black', linewidth=0.5, 
                                 ax=ax, width=0.91)  # Adjusted bar width

    set_common_plot_style('Number of Accidents per Month by Category', 'Month', 'Number of Accidents')
  
    # Create a custom legend and set its title to 'Categories'
    legend_patches = [mpatches.Patch(color=color, label=cat) for cat, color in category_color_map.items()]
    legend = plt.legend(handles=legend_patches, title='Categories', fontsize=14)

    # Increase the font size of the legend title
    plt.setp(legend.get_title(), fontsize=16)

    # Convert month numbers to 3-letter month names
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    ax.set_xticks(range(len(month_names)))  # Ensure there are 12 x-ticks
    ax.set_xticklabels(month_names, rotation=0, fontsize=14)  # Set month names with horizontal alignment

    # Directly setting the font size of the axis labels
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Number of Accidents', fontsize=16)
    
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    save_plot('plot_accidents_per_month.png')
    plt.show()

# Function to create vertical bar graph for categories
def category_bar_graph(df):
    """
    Creates a vertical bar graph showing the number of accidents in each category.

    :param df: DataFrame containing unmanned aviation accident data with categorized accidents.
    :return: None. Displays the bar graph of accidents per category.
    """
    # Use Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    # Prepare the data for plotting
    category_counts = df['First_Category'].value_counts()
    categories = category_counts.index
    counts = category_counts.values
    colors = [category_color_map.get(cat, 'gray') for cat in categories]
    # Create a bar plot using Matplotlib, setting individual colors for each bar
    plt.figure(figsize=(10, 6))
    for i, cat in enumerate(categories):
        plt.bar(cat, counts[i], color=colors[i])
    set_common_plot_style('Number of Accidents in Each Category', 'Category', 'Number of Accidents')
    # Format y-axis as integers
    ax = plt.gca()
    format_axis_as_integers(ax, axis='y')
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    save_plot('category_bar_graph.png')
    plt.show()

# Create an aggregate time series column for accidents per month
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
df.set_index('Date', inplace=True)
accidents_per_month = df.resample('ME').size()
accidents_per_month.name = 'Accidents'

def heatmap_seasonal_trends(df):
    """
    Creates a heatmap to visualize seasonal trends in aviation accidents.
    
    :param df: DataFrame containing unmanned aviation accident data with 'Year' and 'Month' columns.
    :return: None. Displays the heatmap of seasonal trends in aviation accidents.
    """
    # Aggregate data by year and month
    pivot_df = df.pivot_table(values='NtsbNo', index='Month', columns='Year', aggfunc='count')

    # Create and display the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt="g")
    plt.title("Seasonal Trends in Aviation Accidents")
    plt.xlabel("Year")
    plt.ylabel("Month")
    save_plot('heatmap_seasonal_trends.png')
    plt.show()

def trend_analysis(time_series):
    """
    Performs trend analysis using Exponential Smoothing on a time series.

    :param time_series: Pandas Series with time series data.
    :return: None. Displays and saves a plot showing the original data and identified trend.
    """
    model = ExponentialSmoothing(time_series, trend='add')
    fitted_model = model.fit()

    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Original')
    plt.plot(fitted_model.fittedvalues, label='Trend', color='red')
    set_common_plot_style('Trend Analysis', 'Time', 'Number of Accidents')
    plt.legend()
    save_plot('trend_analysis.png')
    plt.show()

def seasonality_analysis(time_series):
    """
    Performs seasonality analysis using Seasonal Decompose on a time series.

    :param time_series: Pandas Series with time series data.
    :return: None. Displays and saves plots for seasonal decomposition.
    """
    decomposition = seasonal_decompose(time_series, model='additive')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8))
    decomposition.trend.plot(ax=ax1, title='Trend')
    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
    decomposition.resid.plot(ax=ax3, title='Residuals')
    plt.tight_layout()
    save_plot('seasonality_analysis.png')
    plt.show()

def categories_and_codes_bar_graph(df):
    """
    Creates a horizontal bar graph showing the Category Names
    and their associated Category Codes.
    """

    sns.set_style("white")

    category_counts = df['First_Category'].value_counts()
    category_codes = category_counts.index
    estimated_max_width = 30  # Adjust as needed
    plt.figure(figsize=(12, 8))
    bars = plt.barh(category_codes, [estimated_max_width] * len(category_codes), color='skyblue')

    for bar, category_code in zip(bars, category_codes):
        category_name = category_mapping.get(category_code, 'Unknown')
        bar_height = bar.get_height()
        y_position = bar.get_y() + bar_height / 2

        # Place the text inside the bar
        x_position = estimated_max_width * 0.04  # Offset from the Y-axis
        plt.text(x_position, y_position, category_name, ha='left', va='center', color='black', 
                 fontsize='large', fontweight='bold')
    plt.xlabel('Category Name')
    plt.ylabel('Category Code')
    plt.title('Category Names and Codes')
    plt.xticks([])

    # Adjust y-axis label spacing
    plt.gca().tick_params(axis='y', which='major', pad=15)  # Increase pad value for more space
    plt.tight_layout()
    save_plot('categories_and_codes_bar_graph.png')
    plt.show()

def radar_chart(df, top_n=5):
    """
    Creates a radar chart comparing the number of accidents for top UAV models across different years.

    Parameters:
    df (DataFrame): DataFrame containing UAV accident data.
    top_n (int): Number of top models to be considered for the chart.

    The function selects the top UAV models based on the number of accidents
    and creates a radar chart to compare the frequency of accidents for each model across different years.
    """

    # Selecting the top N UAV models based on the number of accidents
    top_models = df['Model'].value_counts().nlargest(top_n).index
    data = df[df['Model'].isin(top_models)]

    # Creating a pivot table to aggregate the number of accidents for each model by year
    radar_data = pd.pivot_table(data, index='Year', columns='Model', aggfunc='size', fill_value=0)

    # Number of variables (models) we're plotting
    num_vars = len(radar_data.columns)

    # Create angles for the radar chart axes
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat the first angle to close the circle

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plotting each year's data
    for idx, row in radar_data.iterrows():
        values = row.tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(angles, values, label=f"Year {idx}")  # Plot the line for each year
        ax.fill(angles, values, alpha=0.25)  # Fill the area under the line

    # Setting the labels for each axis
    ax.set_theta_offset(pi / 2)  # Offset the start of the first axis
    ax.set_theta_direction(-1)   # Set the direction of axes labels (clockwise)
    labels = ["Accidents - " + model for model in radar_data.columns.tolist()]
    ax.set_xticks(angles[:-1])   # Set the ticks for each model
    ax.set_xticklabels(labels)   # Set the labels for each model

    # Adding title and subtitle
    plt.title("Comparative Analysis of UAV Accidents Across Top 5 Models by Year", fontsize=16)
    plt.suptitle("Frequency of Accidents per Model Over Different Years", fontsize=12, y=1.05)

    # Adding a descriptive legend in the lower-left
    legend = ax.legend(loc='lower left', bbox_to_anchor=(-0.1, 0.1), title="Accident Data by Year")

    # Adding instructional text in the lower-left
    plt.figtext(0.1, 0.05, "Each axis represents a different UAV model.\n"
                          "Distance from center indicates number of accidents.\n"
                          "Each line corresponds to a different year.",
                horizontalalignment='left', size=12, color="black")
    save_plot('radar_chart.png')
    plt.show()

# Function to convert month number to a 3-letter month abbreviation
def month_num_to_name(month_num):
    """
    Converts a numerical month representation to a 3-letter month name.

    :param month_num: Integer representing the month (1 for January, 2 for February, etc.).
    :return: A 3-letter string representing the month name. Returns an empty string for invalid inputs.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return month_names[month_num - 1] if 1 <= month_num <= 12 else ""

# Function to get a 3-letter country code from country name
def get_3_letter_country_code(country_name):
    """
    Retrieves the 3-letter country code for a given country name.

    :param country_name: Name of the country.
    :return: A 3-letter country code. Returns 'Unknown' if the country is not found.
    """
    return pycountry.countries.get(name=country_name).alpha_3 if pycountry.countries.get(name=country_name) else 'Unknown'

# Apply the function to get country codes and process location details in DataFrame

# Create a mapping dictionary for country codes
country_code_map = {country.name: country.alpha_3 for country in pycountry.countries}
df['CountryCode3'] = df['Country'].map(country_code_map).fillna('Unknown')
# Vectorized string concatenation
df['Location'] = df['City'].str.cat(df[['State', 'CountryCode3']].astype(str), sep=', ', na_rep='').str.strip(', ')

# Define hex color codes corresponding to the folium pin colors for the legend
hex_colors = {
    'red': '#FF0000', 'darkred': '#8B0000', 'lightblue': '#ADD8E6', 'orange': '#FFA500', 
    'darkpurple': '#800080', 'black': '#000000', 'blue': '#0000FF', 'darkgreen': '#006400', 
    'green': '#008000', 'lightgreen': '#90EE90', 'lightgray': '#D3D3D3', 'pink': '#FFC0CB', 
    'gray': '#808080', 'beige': '#F5F5DC', 'lightred': '#FF6347', 'cadetblue': '#5F9EA0', 
    'darkblue': '#00008B', 'purple': '#800080', 'white': '#FFFFFF'
}

# Function to format additional details about accidents
def format_additional_details(row):
    """
    Formats additional details about an accident into a readable string.

    :param row: A DataFrame row containing accident details.
    :return: A string with formatted additional details, separated by HTML line breaks.
    """
    display_names = {
        'Make': 'Make', 'Model': 'Model', 'AirCraftCategory': 'Aircraft Category', 
        'AirportName': 'Airport Name', 'AirCraftDamage': 'Aircraft Damage', 
        'FatalInjuryCount': 'Fatal Injury Count', 'SeriousInjuryCount': 'Serious Injury Count', 
        'MinorInjuryCount': 'Minor Injury Count'
    }
    details = [f"{display_name}: {row[field]}" for field, display_name in display_names.items() 
               if field in row and not pd.isna(row[field]) and (row[field] > 0 if field.endswith('Count') else True)]
    return '<br>'.join(details)

def initialize_map(df):
    # Calculate the geographical center of all accidents for map initialization
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    return folium.Map(location=map_center, zoom_start=5)

def add_markers_to_map(accident_map, df, category_colors):
    # Add formatted additional details to DataFrame
    df['AdditionalDetails'] = df.apply(format_additional_details, axis=1)

    # Iterate through each row in the DataFrame to place markers
    for _, row in df.iterrows():
        # Check for valid latitude and longitude values
        if not np.isnan(row['Latitude']) and not np.isnan(row['Longitude']):
            # Assign a color to the marker based on its category
            marker_color = category_colors.get(row['First_Category'], 'gray')
            # Create a popup text with detailed accident information
            popup_text = f"Report number: {row['NtsbNo']}<br>Location: {row['Location']}<br>Date: {month_num_to_name(row['Month'])}, {row['Year']}<br>Accident Category: {row['First_Category']}<br>{row['AdditionalDetails']}"
            # Place a marker on the map at the accident location
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_text, max_width=450),
                icon=folium.Icon(color=marker_color)
            ).add_to(accident_map)

# Function to create a map with accident markers and a legend
def create_map_with_markers(df, categories_df):
    # Determine the range of years covered by the data
    from_year = df['Year'].min()
    to_year = df['Year'].max()

    # Create title and subtitle for the map using the data range
    map_title = "Interactive Geographical Overview of Unmanned Aerial Accident Reports"
    map_subtitle = f"Based on NTSB Data {from_year} to {to_year}"

    # Create a Folium map centered at the calculated geographical center
    accident_map = initialize_map(df)

    # Define a color palette for the accident markers
    folium_pin_colors = [
        'red', 'darkred', 'orange', 'green', 'darkpurple', 
        'lightgreen', 'black', 'pink', 'blue', 'beige', 'darkgreen', 'lightblue',
        'lightgray', 'cadetblue', 'gray', 'lightred', 'purple', 'darkblue', 'white'
    ]

    # Define hex color codes corresponding to the folium pin colors for the legend
    hex_colors = {
        'red': '#FF0000', 'darkred': '#8B0000', 'lightblue': '#ADD8E6', 'orange': '#FFA500', 
        'darkpurple': '#800080', 'black': '#000000', 'blue': '#0000FF', 'darkgreen': '#006400', 
        'green': '#008000', 'lightgreen': '#90EE90', 'lightgray': '#D3D3D3', 'pink': '#FFC0CB', 
        'gray': '#808080', 'beige': '#F5F5DC', 'lightred': '#FF6347', 'cadetblue': '#5F9EA0', 
        'darkblue': '#00008B', 'purple': '#800080', 'white': '#FFFFFF'
    }
    # Create a dictionary to map category codes to colors
    category_color_map = {}
        # Create a cycle of colors for consistent use in accident markers
    color_cycle = cycle(folium_pin_colors)
    # Map each unique category to a specific color
    unique_categories = df['First_Category'].unique()
    category_colors = {cat: next(color_cycle) for cat in unique_categories}
    # Create a mapping from category codes to more readable category names
    category_name_mapping = dict(zip(df['First_Category'], df['CategoryName']))

    add_markers_to_map(accident_map, df, category_colors)


   # Prepare data for the legend showing category counts
    category_counts = df['First_Category'].value_counts().to_dict()
    sorted_categories = sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True)
    legend_items = ""
    for cat_code in sorted_categories:
        # Set color and count for each category in the legend
        pin_color = category_colors.get(cat_code, 'gray')
        hex_color = hex_colors.get(pin_color, '#808080')
        category_color_map[cat_code] = hex_color
        occurrences = category_counts.get(cat_code, 0)
        cat_name = category_name_mapping.get(cat_code, 'Unknown')
        color_square = f'<i style="background:{hex_color};width:10px;height:10px;display:block;border:1px solid #000;"></i>'
        legend_items += f'<tr><td>{color_square}</td><td>{occurrences}</td><td>{cat_code}</td><td>{cat_name}</td></tr>'

    # Construct the HTML for the legend and add it to the map
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: auto; max-width: 60%; height: auto; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 5px;
                opacity: 0.8;
                overflow: auto;">
      <table style="width:100%;">
        <tr>
          <th>Color</th>
          <th>Count</th>
          <th>Category</th>
          <th>Category Name</th>
        </tr>
        {legend_items}
      </table>
    </div>
    """
    accident_map.get_root().html.add_child(folium.Element(legend_html.format(legend_items=legend_items)))

    # Construct the HTML for the title and subtitle and add it to the map
    title_html = f'''
        <div style="position: fixed; 
                    opacity: 0.8;
                    top: 10px; left: 50px; width: 500px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px;">
            <h3 style="margin:0;">{map_title}</h3>
            <h4 style="margin:0;">{map_subtitle}</h4>
        </div> 
     '''
    accident_map.get_root().html.add_child(folium.Element(title_html))

    # Return the final map object with all markers, legend, and titles
    # Return the category color map
    return accident_map, category_color_map


if __name__ == "__main__":
    try:
        # Create and save the map with accident markers and legend.
        # Generate an interactive map with markers for each accident and a legend using the provided data
        # The map is created using the Folium library, and it visualizes the geographical distribution of accidents
        accident_map, category_color_map = create_map_with_markers(df, categories_df)

        # Save the generated interactive map as an HTML file.
        # The saved map can be viewed in any web browser and allows interactive exploration of the accident data
        accident_map.save('c:/NTSB/output/accident_map.html')

        # Running the visualization functions

        # Plot and display the number of unmanned aviation accidents per year
        # This function generates a line chart visualizing the trend of accidents over the years
        plot_accidents_per_year(df)

        # Plot and display the number of unmanned aviation accidents per month
        # This function creates a line chart showing the distribution of accidents across different months
        plot_accidents_per_month(df)

        # Generate and display a bar graph showing the number of accidents in each category
        # This bar graph provides a visual comparison of accident frequencies across different categories
        category_bar_graph(df)
        
        # Create a horizontal bar graph showing the Category Names
        # and their associated Category Codes
        categories_and_codes_bar_graph (df)

        # Create aggregated time series
        accidents_per_month = df.resample('ME').size()
        accidents_per_month.name = 'Accidents'

        # Create a heatmap to visualize seasonal trends in aviation accidents
        heatmap_seasonal_trends(df)

        # Time Series Analysis
        trend_analysis(accidents_per_month)
        seasonality_analysis(accidents_per_month)

        # A radar chart comparing the number of accidents for top UAV models across different years.
        radar_chart(df, top_n=5)
        
        logging.info("Script executed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File operation error: {e}")
    except pd.errors.ParserError as e:
        logging.error(f"Data parsing error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise