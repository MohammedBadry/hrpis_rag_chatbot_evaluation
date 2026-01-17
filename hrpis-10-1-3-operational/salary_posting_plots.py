# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 01:20:13 2025

@author: Firas.Alhawari
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- File path ---
file_path = r"./salary_posting.csv"

# --- Load CSV ---
df = pd.read_csv(file_path)

# --- Parse start/end dates ---
date_format = '%d-%b-%y %I.%M.%S.%f %p'
df['MIN_DATE_BULK'] = pd.to_datetime(df['MIN_DATE_BULK'], format=date_format, errors='coerce')
df['MAX_DATE_BULK'] = pd.to_datetime(df['MAX_DATE_BULK'], format=date_format, errors='coerce')

# --- User control: Start month ---
start_year = 2024
start_month = 2
end_year = 2025
end_month = 8

# --- Filter based on user-controlled start month ---
df = df[
    ((df['YEAR'] > start_year) | ((df['YEAR'] == start_year) & (df['MONTH'] >= start_month))) &
    ((df['YEAR'] < end_year) | ((df['YEAR'] == end_year) & (df['MONTH'] <= end_month)))
]

# --- Month-year for x-axis ---
df['month_year'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')

# --- Compute start/end relative to month (days) ---
df['start_day'] = (df['MIN_DATE_BULK'] - df['month_year']).dt.total_seconds() / 86400
df['end_day'] = (df['MAX_DATE_BULK'] - df['month_year']).dt.total_seconds() / 86400

# --- Corrected cycle duration in minutes ---
df['cycle_minutes'] = df['DURATION_MINUTES_BULK']

# --- Generate full month range for x-axis ---
all_months = pd.date_range(start=f'{start_year}-{start_month:02d}-01', end=f'{end_year}-{end_month:02d}-01', freq='MS')
df_all = pd.DataFrame({'month_year': all_months})
df_plot = pd.merge(df_all, df, on='month_year', how='left')

# --- White background, dark ticks ---
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black'
})

# --- Plot 1: Number of slips per month ---
fig1, ax1 = plt.subplots(figsize=(12,5))
ax1.scatter(df_plot['month_year'], df_plot['BULK_COUNT'], color='blue', s=80)
ax1.set_title('Number of Bulk Salary Slips Processed per Month', fontsize=14)
ax1.set_xlabel('Month', fontsize=13)
ax1.set_ylabel('Employee Count', fontsize=13)
ax1.set_ylim(bottom=0)
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.setp(ax1.get_xticklabels(), rotation=45, fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
fig1.tight_layout()
fig1.savefig('plot_slips_per_month.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# --- Plot 2: Start and end of bulk ---
fig2, ax2 = plt.subplots(figsize=(12,5))
jitter = 0.2
ax2.scatter(df_plot['month_year'], df_plot['start_day'] + np.random.uniform(-jitter, jitter, len(df_plot)), color='green', label='Start Day', s=80)
ax2.scatter(df_plot['month_year'], df_plot['end_day'] + np.random.uniform(-jitter, jitter, len(df_plot)), color='red', label='End Day', s=80)
ax2.set_title('Start and End Days of Monthly Bulk Salary Slip Processing', fontsize=14)
ax2.set_xlabel('Month', fontsize=13)
ax2.set_ylabel('Day of Month', fontsize=13)
ax2.set_ylim(bottom=0, top=31)
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)
fig2.tight_layout()
fig2.savefig('plot_bulk_start_end.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# --- Plot 3: Cycle duration with outlier inset ---
bulk_mask = df_plot['cycle_minutes'] < 15
outlier_mask = df_plot['cycle_minutes'] >= 15

fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.scatter(df_plot.loc[bulk_mask, 'month_year'], df_plot.loc[bulk_mask, 'cycle_minutes'], color='orange', s=80, label='Bulk (<15 min)')
ax3.set_title('Bulk Salary Slip Generation Time for All Employees per Month', fontsize=14)
ax3.set_xlabel('Month', fontsize=13)
ax3.set_ylabel('Duration (minutes)', fontsize=13)
ax3.set_ylim(bottom=0, top=15)
ax3.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax3.xaxis.set_major_locator(mdates.MonthLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.setp(ax3.get_xticklabels(), rotation=45, fontsize=12)
plt.setp(ax3.get_yticklabels(), fontsize=12)

# --- Inset for outliers ---
axins = inset_axes(ax3, width="40%", height="40%", loc='upper center')
axins.scatter(df_plot.loc[outlier_mask, 'month_year'], df_plot.loc[outlier_mask, 'cycle_minutes'], color='red', s=50, label='Outliers (>=15 min)')

# Y-padding
ymin = df_plot.loc[outlier_mask, 'cycle_minutes'].min()
ymax = df_plot.loc[outlier_mask, 'cycle_minutes'].max()
ypad = (ymax - ymin) * 0.15 if ymax != ymin else 5
axins.set_ylim(ymin - ypad, ymax + ypad)

# X-padding
xmin = df_plot.loc[outlier_mask, 'month_year'].min()
xmax = df_plot.loc[outlier_mask, 'month_year'].max()
xpad = pd.Timedelta(days=15)
axins.set_xlim(xmin - xpad, xmax + xpad)

# X-axis formatting
axins.xaxis.set_major_locator(mdates.MonthLocator())
axins.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
axins.tick_params(axis='x', rotation=45, labelsize=12)
axins.tick_params(axis='y', labelsize=12)
axins.grid(True, linestyle='--', alpha=0.5, color='gray')
axins.legend(fontsize=10, loc='upper left')

fig3.tight_layout()
fig3.savefig('plot_cycle_duration_with_outliers.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

