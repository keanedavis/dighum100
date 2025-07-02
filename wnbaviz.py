import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="WNBA Attendance & Trends Dashboard", page_icon="🏀")

# --- Define a custom color palette for teams (using Plotly's qualitative colors) ---
# This ensures a variety of distinct colors for different teams
TEAM_COLORS = px.colors.qualitative.Alphabet + px.colors.qualitative.G10 + px.colors.qualitative.D3

# Function to assign a unique color to each team
def get_team_color_map(teams):
    color_map = {}
    for i, team in enumerate(teams):
        color_map[team] = TEAM_COLORS[i % len(TEAM_COLORS)] # Cycle through available colors
    return color_map

# --- Load Attendance Data ---
@st.cache_data # Cache data to improve performance
def load_attendance_data(file_path):
    """
    Loads the WNBA attendance data from a CSV file.
    Performs basic data cleaning: ensures Attendance is numeric and creates a datetime column.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert 'Attendance' to numeric, coercing errors to NaN
        df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')
        
        # Drop rows where Attendance is NaN (if any non-numeric values were present)
        df.dropna(subset=['Attendance'], inplace=True)
        
        # --- Robust Date Creation ---
        # Ensure 'Day' and 'Year' are numeric first, coercing errors
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        # Drop rows where Year, Month, or Day are missing after initial loading,
        # as these are critical for date creation.
        # Assuming 'Month' column exists as either number or month name string.
        df.dropna(subset=['Year', 'Month', 'Day'], inplace=True)

        # Convert 'Year' and 'Day' to integer types after dropping NaNs
        df['Year'] = df['Year'].astype(int)
        df['Day'] = df['Day'].astype(int)

        # Create a combined date string in 'YYYY-MonthName-DD' or 'YYYY-MM-DD' format.
        # pandas.to_datetime is smart enough to parse various month formats (names or numbers).
        df['Date_Str'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str)
        df['Date'] = pd.to_datetime(df['Date_Str'], errors='coerce')
        
        # Drop rows where date conversion failed (e.g., invalid date combinations)
        df.dropna(subset=['Date'], inplace=True) 
        # --- End Robust Date Creation ---

        # Create 'MonthYear' for monthly aggregation and 'DayOfWeek' for insights
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df['DayOfWeek'] = df['Date'].dt.day_name() # Get day name (Monday, Tuesday, etc.)
        df['MonthName'] = df['Date'].dt.strftime('%b') # Get short month name (Jan, Feb, etc.)

        # Sort by date for better time-series plotting
        df = df.sort_values(by='Date')

        return df
    except FileNotFoundError:
        st.error(f"Error: The attendance file '{file_path}' was not found. Please make sure it's in the same directory as this script.")
        return pd.DataFrame() # Return an empty DataFrame on error
    except Exception as e:
        st.error(f"An error occurred while loading or processing the attendance data: {e}")
        return pd.DataFrame()

# --- Load Media Data ---
@st.cache_data # Cache data to improve performance
def load_media_data(file_path):
    """
    Loads the media coverage data from a CSV file.
    Uses 'publish_date' for dates and infers 'mentions' by counting rows.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure 'publish_date' is datetime
        if 'publish_date' in df.columns:
            df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
            df.dropna(subset=['publish_date'], inplace=True)
            df.rename(columns={'publish_date': 'Date'}, inplace=True) # Align column name
        else:
            st.warning("Column 'publish_date' not found in media data. Please ensure the CSV has a 'publish_date' column.")
            return pd.DataFrame()

        # Infer mentions by counting rows, useful if no explicit 'Mentions' column
        # Each row is assumed to be one mention/article
        df['Mentions'] = 1 
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df = df.sort_values(by='Date')

        return df
    except FileNotFoundError:
        st.error(f"Error: The media file '{file_path}' was not found. Please make sure it's in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading or processing the media data: {e}")
        return pd.DataFrame()

# --- Load Google Search Trend Data ---
@st.cache_data # Cache data to improve performance
def load
