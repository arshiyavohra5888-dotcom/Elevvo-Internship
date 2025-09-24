 # ==============================
## Task 3: Sales Forecasting Implementation -Elevvo Tech Internship 
# ==============================

# ------------------------------
# 1. Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model and Evaluation Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Advanced Modeling
import xgboost as xgb
import lightgbm as lgb

# Statistical Analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ------------------------------
# 2. Enhanced Styling Configuration
# ------------------------------
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Elegant purple
    'accent': '#F18F01',       # Warm orange
    'success': '#C73E1D',      # Deep red
    'background': '#FAFAFA',   # Clean light gray
    'text': '#2C3E50',         # Dark blue-gray
    'grid': '#E8E8E8',         # Light grid
    'highlight': '#FFD93D'     # Bright yellow
}

# Enhanced matplotlib configuration
plt.rcParams.update({
    'figure.figsize': [16, 10],
    'figure.facecolor': 'white',
    'axes.facecolor': COLORS['background'],
    'axes.edgecolor': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'font.size': 11,
    'font.family': 'Segoe UI',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.titlepad': 20,
    'axes.labelpad': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

# ------------------------------
# 3. Define Sales Forecast Class
# ------------------------------
class SalesForecaster:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()

    def generate_sample_data(self):
        np.random.seed(42)
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
        base_sales = 1000
        trend = np.linspace(0, 800, len(dates))
        seasonal = 300 * np.sin(2 * np.pi * dates.dayofyear / 365)
        weekly_pattern = 150 * np.sin(2 * np.pi * dates.dayofweek / 7)

        holiday_effect = np.zeros(len(dates))
        holiday_months = [11, 12]
        holiday_days = [24, 25, 26, 31, 1]
        for i, date in enumerate(dates):
            if date.month in holiday_months and date.day in holiday_days:
                holiday_effect[i] = 500
            elif date.month == 7 and date.day == 4:
                holiday_effect[i] = 300

        sales = (base_sales + trend + seasonal + weekly_pattern +
                 holiday_effect + np.random.normal(0, 80, len(dates)))

        stores = ['Store_1', 'Store_2', 'Store_3', 'Store_4', 'Store_5']
        depts = ['Dept_A', 'Dept_B', 'Dept_C', 'Dept_D']

        data = []
        for store in stores:
            store_multiplier = np.random.uniform(0.7, 1.3)
            for dept in depts:
                dept_multiplier = np.random.uniform(0.8, 1.2)
                adjusted_sales = sales * store_multiplier * dept_multiplier
                for i, date in enumerate(dates):
                    if np.random.random() > 0.98:
                        continue
                    data.append({
                        'Date': date,
                        'Store': store,
                        'Dept': dept,
                        'Weekly_Sales': max(100, adjusted_sales[i]),
                        'Temperature': np.random.normal(18, 12),
                        'Fuel_Price': np.random.uniform(2.2, 4.5),
                        'CPI': np.random.uniform(160, 280),
                        'Unemployment': np.random.uniform(3.5, 9.0),
                        'IsHoliday': int(date.month in holiday_months and date.day in holiday_days or
                                         (date.month == 7 and date.day == 4))
                    })
        df = pd.DataFrame(data)
        return df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

    def create_time_features(self, df):
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        df['Season'] = df['Month'] % 12 // 3 + 1
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        return df

    def create_lag_features(self, df, lags=[1, 2, 3, 7, 14, 30, 60, 90, 365]):
        df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        for lag in lags:
            df[f'Lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
        for window in [7, 14, 30, 60, 90]:
            df[f'Rolling_Mean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'Rolling_Std_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
            df[f'Rolling_Min_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            df[f'Rolling_Max_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
        for span in [7, 14, 30]:
            df[f'EMA_{span}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
                lambda x: x.ewm(span=span).mean())
        return df

    def seasonal_decomposition_analysis(self, df, store='Store_1', dept='Dept_A'):
        """Create an elegant seasonal decomposition visualization with no overlapping"""
        store_dept_data = df[(df['Store'] == store) & (df['Dept'] == dept)]
        weekly_data = store_dept_data.set_index('Date')['Weekly_Sales'].resample('W').mean()
        decomposition = seasonal_decompose(weekly_data.dropna(), model='additive', period=52)

        # Create figure with better layout - wider and taller
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        fig.suptitle(f'Seasonal Decomposition Analysis\n{store} - {dept}',
                     fontsize=20, fontweight='bold', color=COLORS['text'],
                     y=0.96)

        # Create gridspec with better spacing
        gs = fig.add_gridspec(4, 1, hspace=0.3, top=0.85, bottom=0.1, left=0.12, right=0.88)

        # Component titles and colors
        components = [
            (weekly_data, 'Original Sales Trend', COLORS['primary'], 'Weekly Sales ($)'),
            (decomposition.trend, 'Long-term Trend', COLORS['secondary'], 'Trend Component'),
            (decomposition.seasonal, 'Seasonal Pattern', COLORS['accent'], 'Seasonal Component'),
            (decomposition.resid, 'Residuals (Noise)', COLORS['success'], 'Residuals')
        ]

        for i, (data, title, color, ylabel) in enumerate(components):
            ax = fig.add_subplot(gs[i])

            # Plot with enhanced styling
            ax.plot(data.index, data.values, color=color, linewidth=2.5, alpha=0.9)
            ax.fill_between(data.index, data.values, alpha=0.1, color=color)

            # Enhanced title with icon
            title_with_icon = f"{'📊' if i == 0 else '📈' if i == 1 else '🌊' if i == 2 else '🎯'} {title}"
            ax.set_title(title_with_icon, fontsize=13, fontweight='bold',
                        color=color, pad=15)

            # Styling
            ax.set_facecolor(COLORS['background'])
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.8)
            ax.set_ylabel(ylabel, fontsize=10, fontweight='600', color=COLORS['text'])

            # Format x-axis dates - only show on bottom plot
            if i == 3:
                ax.tick_params(axis='x', rotation=20, labelsize=9)
                ax.set_xlabel('Date', fontsize=10, fontweight='600', color=COLORS['text'])
            else:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)

            # Add value annotations for key points - positioned better
            if len(data) > 0:
                max_idx = data.idxmax()
                min_idx = data.idxmin()
                max_val = data.max()
                min_val = data.min()

                # Position annotations to avoid overlap
                if i == 0:  # Original data
                    ax.annotate(f'Peak: ${max_val",.0f"}',
                               xy=(max_idx, max_val),
                               xytext=(15, 15), textcoords='offset points',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

                    ax.annotate(f'Min: ${min_val",.0f"}',
                               xy=(min_idx, min_val),
                               xytext=(15, -20), textcoords='offset points',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
                elif i == 1:  # Trend
                    ax.annotate(f'Trend Peak: ${max_val",.0f"}',
                               xy=(max_idx, max_val),
                               xytext=(-60, 15), textcoords='offset points',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

        # Add summary statistics box - positioned better
        stats_text = f"""Summary Statistics:
• Mean: ${weekly_data.mean()",.0f"}
• Std Dev: ${weekly_data.std()",.0f"}
• Min: ${weekly_data.min()",.0f"}
• Max: ${weekly_data.max()",.0f"}
• Observations: {len(weekly_data)","}"""

        fig.text(0.02, 0.02, stats_text,
                fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom')

        # Add interpretation box
        interp_text = """Interpretation Guide:
• Original: Raw sales data
• Trend: Long-term direction
• Seasonal: Repeating patterns
• Residual: Random variation"""

        fig.text(0.98, 0.02, interp_text,
                fontsize=8, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='lightgreen', alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right')

        plt.tight_layout()
        plt.show()
        return decomposition
