# santa-clara-county-school-absenteeism-analysis
Analysis of chronic absenteeism in Santa Clara County schools
# Santa Clara County School Chronic Absenteeism Analysis

## Overview
Analysis of chronic absenteeism rates across Santa Clara County schools for the 2023-2024 academic year. This project provides comprehensive data visualizations and statistical analysis to identify patterns, trends, and schools requiring intervention.

## Project Description
This project analyzes chronic absenteeism data from California schools with a focus on Santa Clara County school-level data. The analysis includes distribution patterns, district comparisons, and risk categorization to support data-driven educational policy decisions.

## Data Source
- **Dataset**: California Department of Education Chronic Absenteeism Data (2023-24)
- **File**: `chronicabsenteeism24.csv` (not included in repository - see Data Setup below)
- **Scope**: Santa Clara County schools only
- **Level**: School-level data (excludes district/county aggregates)
- **Source URL**: [California Department of Education Data](https://www.cde.ca.gov/ds/ad/filesabd.asp)

## Features
- **Distribution Analysis**: Multiple histogram types with statistical overlays
- **District Comparisons**: Side-by-side analysis of all Santa Clara County school districts
- **Risk Categorization**: Schools grouped by absenteeism severity levels
- **Statistical Summaries**: Comprehensive descriptive statistics with outlier detection
- **Visualizations**: Box plots, cumulative distribution functions, and color-coded charts
- **Category Analysis**: Breakdown by school type (charter vs. traditional) and grade levels

## Requirements
```
pandas>=1.5.0
numpy>=1.20.0
seaborn>=0.11.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## Installation
1. Clone this repository:
```bash
git clone https://github.com/jesskw21/santa-clara-county-school-absenteeism-analysis.git
cd santa-clara-county-school-absenteeism-analysis
```

2. Install required packages:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

3. Download and setup data (see Data Setup section below)

4. Run the analysis:
```bash
python school_prediction_engine.py
```

## Data Setup
**Important**: The CSV data file is not included in this repository to protect sensitive school information.

1. Download the 2023-24 chronic absenteeism data from the California Department of Education
2. Save the file as `chronicabsenteeism24.csv` in the project directory
3. Ensure your CSV contains these required columns:
   - `County Name`
   - `District Name`
   - `School Name`
   - `Aggregate Level`
   - `ChronicAbsenteeismRate`
   - `Charter School`
   - `Reporting Category`

## Usage
Run the main analysis script:
```python
python school_prediction_engine.py
```

The script will automatically:
- Load and filter Santa Clara County school-level data
- Generate comprehensive visualizations
- Provide statistical analysis and summaries
- Create district-by-district comparisons
- Output key findings to the console

## Output and Visualizations
The analysis generates several types of visualizations:

1. **Distribution Analysis**:
   - Basic histogram with mean overlay
   - Density plots
   - Box plots with quartile information
   - Cumulative distribution functions

2. **Detailed Analysis**:
   - Color-coded histogram by risk levels
   - Statistical markers (mean, median, standard deviation)
   - Risk category breakdown

3. **Comparative Analysis**:
   - District-by-district box plots
   - Charter vs. traditional school comparisons
   - Grade level analysis

## Key Functions
- `load_and_explore_data()`: Initial data loading and exploration
- `filter_santa_clara_school_data()`: Filters for county and school-level records
- `clean_data()`: Data preprocessing and outlier removal
- `create_histogram_visualizations()`: Generates comprehensive distribution plots
- `create_detailed_histogram()`: Creates risk-categorized visualization
- `analyze_santa_clara_districts()`: District-level comparative analysis
- `analyze_by_categories()`: Breakdown by school characteristics

## Key Findings
The analysis provides insights into:
- Distribution of chronic absenteeism rates across Santa Clara County schools
- Identification of districts and schools with highest/lowest rates
- Comparison between charter and traditional public schools
- Grade-level patterns in absenteeism rates

## Applications
This analysis supports:
- **School Board Presentations**: Data-driven policy recommendations
- **Resource Allocation**: Identifying schools needing intervention
- **District Planning**: Comparative performance analysis
- **Educational Research**: Understanding absenteeism patterns

## Data Privacy
- Raw school data is not included in this repository
- Analysis focuses on publicly available aggregate statistics
- Individual student data is not accessed or analyzed

## Contributing
This project was developed for educational policy research. Contributions and adaptations for other counties or analysis approaches are welcome.

## License
MIT License - See LICENSE file for details

## Acknowledgments
- California Department of Education for providing public education data
- Santa Clara County Office of Education
- Milpitas Unified School District (primary focus area)

## Contact
For questions about the analysis methodology or findings, please open an issue in this repository.
