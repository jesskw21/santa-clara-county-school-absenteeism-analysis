import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the chronic absenteeism data and explore its structure"""
    print("Loading chronic absenteeism data...")
    df = pd.read_csv('chronicabsenteeism24.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def filter_santa_clara_school_data(df):
    """Filter data for Santa Clara County school-level data only"""
    print("\nFiltering for Santa Clara County school-level data...")
    
    # Filter for Santa Clara County
    santa_clara_data = df[df['County Name'] == 'Santa Clara'].copy()
    print(f"Santa Clara County data shape: {santa_clara_data.shape}")
    
    # Filter for school-level data only (Aggregate Level = 'S')
    santa_clara_schools = santa_clara_data[santa_clara_data['Aggregate Level'] == 'S'].copy()
    print(f"Santa Clara County school-level data shape: {santa_clara_schools.shape}")
    
    # Show unique districts and schools
    print(f"Number of unique districts: {santa_clara_schools['District Name'].nunique()}")
    print(f"Number of unique schools: {santa_clara_schools['School Name'].nunique()}")
    
    print("\nSample of Santa Clara County school data:")
    print(santa_clara_schools[['District Name', 'School Name', 'Charter School', 'ChronicAbsenteeismRate']].head(10))
    
    return santa_clara_schools

def clean_data(df):
    """Clean and preprocess the data"""
    print("\nData cleaning and preprocessing...")
    
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Remove rows with missing chronic absenteeism rate
    original_count = len(df)
    df_clean = df.dropna(subset=['ChronicAbsenteeismRate'])
    print(f"Removed {original_count - len(df_clean)} rows with missing ChronicAbsenteeismRate")
    
    # Convert rate to numeric if it's not already
    df_clean['ChronicAbsenteeismRate'] = pd.to_numeric(df_clean['ChronicAbsenteeismRate'], errors='coerce')
    
    # Remove any remaining rows with invalid rate values
    df_clean = df_clean.dropna(subset=['ChronicAbsenteeismRate'])
    
    # Remove outliers (rates > 100% are likely data errors)
    df_clean = df_clean[df_clean['ChronicAbsenteeismRate'] <= 100]
    
    print(f"Final dataset shape after cleaning: {df_clean.shape}")
    
    return df_clean

def create_histogram_visualizations(df):
    """Create comprehensive histogram visualizations for chronic absenteeism rate"""
    
    # Statistical summary
    print("\nChronic Absenteeism Rate Statistics:")
    print(df['ChronicAbsenteeismRate'].describe())
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Santa Clara County Schools - Chronic Absenteeism Rate Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Basic histogram
    axes[0, 0].hist(df['ChronicAbsenteeismRate'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Basic Histogram of Chronic Absenteeism Rate')
    axes[0, 0].set_xlabel('Chronic Absenteeism Rate (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add mean line
    mean_rate = df['ChronicAbsenteeismRate'].mean()
    axes[0, 0].axvline(mean_rate, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rate:.1f}%')
    axes[0, 0].legend()
    
    # 2. Density plot (histogram with density)
    axes[0, 1].hist(df['ChronicAbsenteeismRate'], bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Density Plot of Chronic Absenteeism Rate')
    axes[0, 1].set_xlabel('Chronic Absenteeism Rate (%)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot
    axes[1, 0].boxplot(df['ChronicAbsenteeismRate'], patch_artist=True, 
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[1, 0].set_title('Box Plot of Chronic Absenteeism Rate')
    axes[1, 0].set_ylabel('Chronic Absenteeism Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_rates = np.sort(df['ChronicAbsenteeismRate'])
    cumulative_prob = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
    axes[1, 1].plot(sorted_rates, cumulative_prob, linewidth=2, color='purple')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].set_xlabel('Chronic Absenteeism Rate (%)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_detailed_histogram(df):
    """Create a more detailed histogram with additional analysis"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram with more bins for better detail
    n, bins, patches = ax.hist(df['ChronicAbsenteeismRate'], bins=100, alpha=0.7, 
                               color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Color bars based on value ranges
    for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
        bin_center = (bin_left + bin_right) / 2
        if bin_center < 10:
            patch.set_facecolor('green')  # Low absenteeism
        elif bin_center < 20:
            patch.set_facecolor('yellow')  # Moderate absenteeism
        elif bin_center < 30:
            patch.set_facecolor('orange')  # High absenteeism
        else:
            patch.set_facecolor('red')  # Very high absenteeism
    
    # Add statistical lines
    mean_rate = df['ChronicAbsenteeismRate'].mean()
    median_rate = df['ChronicAbsenteeismRate'].median()
    std_rate = df['ChronicAbsenteeismRate'].std()
    
    ax.axvline(mean_rate, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rate:.1f}%')
    ax.axvline(median_rate, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_rate:.1f}%')
    ax.axvline(mean_rate - std_rate, color='gray', linestyle=':', alpha=0.7, label=f'-1 Std: {mean_rate - std_rate:.1f}%')
    ax.axvline(mean_rate + std_rate, color='gray', linestyle=':', alpha=0.7, label=f'+1 Std: {mean_rate + std_rate:.1f}%')
    
    ax.set_title('Santa Clara County Schools - Detailed Histogram of Chronic Absenteeism Rate Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Chronic Absenteeism Rate (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with statistics
    stats_text = f'''Statistics:
    Count: {len(df):,}
    Mean: {mean_rate:.2f}%
    Median: {median_rate:.2f}%
    Std Dev: {std_rate:.2f}%
    Min: {df['ChronicAbsenteeismRate'].min():.2f}%
    Max: {df['ChronicAbsenteeismRate'].max():.2f}%'''
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_by_categories(df):
    """Analyze chronic absenteeism rates by different categories"""
    
    print("\n" + "="*50)
    print("ANALYSIS BY CATEGORIES")
    print("="*50)
    
    # Analysis by grade level
    if 'Reporting Category' in df.columns:
        print("\nChronic Absenteeism Rate by Grade Level:")
        grade_analysis = df.groupby('Reporting Category')['ChronicAbsenteeismRate'].agg(['count', 'mean', 'median', 'std']).round(2)
        print(grade_analysis)
        
        # Create box plot by grade level
        plt.figure(figsize=(12, 6))
        df.boxplot(column='ChronicAbsenteeismRate', by='Reporting Category', ax=plt.gca())
        plt.title('Chronic Absenteeism Rate by Grade Level')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Analysis by charter vs non-charter schools
    if 'Charter School' in df.columns:
        print("\nChronic Absenteeism Rate by School Type:")
        charter_analysis = df.groupby('Charter School')['ChronicAbsenteeismRate'].agg(['count', 'mean', 'median', 'std']).round(2)
        print(charter_analysis)
        
        # Create box plot by school type
        plt.figure(figsize=(8, 6))
        df.boxplot(column='ChronicAbsenteeismRate', by='Charter School', ax=plt.gca())
        plt.title('Chronic Absenteeism Rate by School Type')
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.show()

def analyze_santa_clara_districts(df):
    """Analyze chronic absenteeism rates by district in Santa Clara County"""
    
    print("\n" + "="*50)
    print("SANTA CLARA COUNTY DISTRICT ANALYSIS")
    print("="*50)
    
    # District-level analysis
    district_analysis = df.groupby('District Name')['ChronicAbsenteeismRate'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2).sort_values('mean', ascending=False)
    
    print("\nChronic Absenteeism Rate by District (sorted by mean rate):")
    print(district_analysis)
    
    # Create box plot by district
    plt.figure(figsize=(15, 8))
    df.boxplot(column='ChronicAbsenteeismRate', by='District Name', ax=plt.gca())
    plt.title('Santa Clara County Schools - Chronic Absenteeism Rate by District')
    plt.suptitle('')  # Remove default title
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Top and bottom performing districts
    print("\nTop 5 Districts with Highest Chronic Absenteeism Rates:")
    print(district_analysis.head())
    
    print("\nTop 5 Districts with Lowest Chronic Absenteeism Rates:")
    print(district_analysis.tail())
    
    return district_analysis

def main():
    """Main function to run the complete analysis for Santa Clara County school-level data"""
    print("CHRONIC ABSENTEEISM RATE DATA VISUALIZATION - SANTA CLARA COUNTY SCHOOLS")
    print("="*70)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Filter for Santa Clara County school-level data
    santa_clara_schools = filter_santa_clara_school_data(df)
    
    # Clean data
    df_clean = clean_data(santa_clara_schools)
    
    # Create visualizations
    print("\nCreating histogram visualizations for Santa Clara County schools...")
    create_histogram_visualizations(df_clean)
    
    print("\nCreating detailed histogram...")
    create_detailed_histogram(df_clean)
    
    # Analyze by categories
    analyze_by_categories(df_clean)
    
    # Analyze Santa Clara County districts
    district_analysis = analyze_santa_clara_districts(df_clean)
    
    print("\nAnalysis complete!")
    
    return df_clean, district_analysis

if __name__ == "__main__":
    df, district_analysis = main()
