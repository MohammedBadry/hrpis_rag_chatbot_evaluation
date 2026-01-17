# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:03:16 2025

@author: Firas.Alhawari
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# --- File paths ---
base_path = r"./"
income_file = os.path.join(base_path, 'income-tax-helpdesk.csv')
salary_file = os.path.join(base_path, 'salary-slip-helpdesk.csv')

# --- Employee count for normalization ---
num_employees = 750  # Fixed average employees

# --- Robust date parsing ---
def parse_dates(df, date_col='Creation Date'):
    formats = ['%d/%m/%Y %H:%M', '%d/%m/%Y %H:%M:%S',
               '%d-%m-%Y %H:%M:%S', '%d-%m-%Y']
    parsed_dates = []
    for fmt in formats:
        parsed = pd.to_datetime(df[date_col], format=fmt,
                                errors='coerce', dayfirst=True)
        parsed_dates.append(parsed)
    df['Creation Date'] = pd.concat(parsed_dates, axis=1).bfill(axis=1).iloc[:, 0]
    df = df.dropna(subset=['Creation Date'])
    return df

# --- Load and preprocess CSV ---
def load_helpdesk_csv(file_path):
    df = pd.read_csv(file_path)
    df = parse_dates(df)
    df['month_year'] = df['Creation Date'].dt.to_period('M').dt.to_timestamp()
    counts = df.groupby('month_year').size().reset_index(name='Tickets')
    return counts

income_counts = load_helpdesk_csv(income_file)
salary_counts = load_helpdesk_csv(salary_file)

# --- User-controlled date range ---
start_date = pd.Timestamp('2024-05-01')
end_date   = pd.Timestamp('2025-09-30')

income_counts = income_counts[(income_counts['month_year'] >= start_date) &
                              (income_counts['month_year'] <= end_date)]
salary_counts = salary_counts[(salary_counts['month_year'] >= start_date) &
                              (salary_counts['month_year'] <= end_date)]

# --- Merge month-year index ---
all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
df_plot = pd.DataFrame({'month_year': all_months})
df_plot = pd.merge(df_plot,
                   income_counts.rename(columns={'Tickets': 'IncomeTaxTickets'}),
                   on='month_year', how='left')
df_plot = pd.merge(df_plot,
                   salary_counts.rename(columns={'Tickets': 'SalarySlipTickets'}),
                   on='month_year', how='left')
df_plot.fillna(0, inplace=True)

# --- Convert to matplotlib date numbers ---
df_plot['month_num'] = mdates.date2num(df_plot['month_year'])

# --- Set bar width and spacing (NO OVERLAP) ---
bar_width = 10        # width of each bar
gap = 0               # gap between the two bars in each pair

income_pos = df_plot['month_num'] - (bar_width/2 + gap)
salary_pos = df_plot['month_num'] + (bar_width/2 + gap)

# --------------------------------------------------------------
# --- Rolling 3-month Standard Deviation for RAW COUNTS ---
# --------------------------------------------------------------
df_plot['IncomeSTD'] = df_plot['IncomeTaxTickets'].rolling(window=3).std()
df_plot['SalarySTD'] = df_plot['SalarySlipTickets'].rolling(window=3).std()

# --------------------------------------------------------------
# --- Plot 1: Raw number of tickets ---
# --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))

ax.bar(income_pos, df_plot['IncomeTaxTickets'], width=bar_width,
       color='skyblue', label='Income Tax Slip Tickets', edgecolor='black')

ax.bar(salary_pos, df_plot['SalarySlipTickets'], width=bar_width,
       color='lightgreen', label='Official Salary Slip Tickets', edgecolor='black')

# Trend lines
ax.plot(df_plot['month_num'], df_plot['IncomeTaxTickets'],
        color='blue', linestyle='--', linewidth=1)
ax.plot(df_plot['month_num'], df_plot['SalarySlipTickets'],
        color='green', linestyle='--', linewidth=1)

# Error bars
ax.errorbar(income_pos,
            df_plot['IncomeTaxTickets'],
            yerr=df_plot['IncomeSTD'],
            fmt='none', ecolor='blue', elinewidth=1.2, capsize=6)

ax.errorbar(salary_pos,
            df_plot['SalarySlipTickets'],
            yerr=df_plot['SalarySTD'],
            fmt='none', ecolor='green', elinewidth=1.2, capsize=6)

# Labels
ax.set_title('Monthly Helpdesk Tickets (Raw Counts)', fontsize=16)
ax.set_xlabel('Month', fontsize=15)
ax.set_ylabel('Number of Tickets', fontsize=15)
ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.setp(ax.get_xticklabels(), rotation=45, fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(base_path, 'helpdesk_tickets_raw.png'),
            dpi=300, bbox_inches='tight', facecolor='white')

# --------------------------------------------------------------
# --- Plot 2: Tickets per 100 employees ---
# --------------------------------------------------------------
df_plot['IncomePer100'] = df_plot['IncomeTaxTickets'] / num_employees * 100
df_plot['SalaryPer100'] = df_plot['SalarySlipTickets'] / num_employees * 100

df_plot['IncomeSTD100'] = df_plot['IncomePer100'].rolling(window=3).std()
df_plot['SalarySTD100'] = df_plot['SalaryPer100'].rolling(window=3).std()

fig2, ax2 = plt.subplots(figsize=(14, 7))

ax2.bar(income_pos, df_plot['IncomePer100'], width=bar_width,
        color='skyblue', label='Income Tax Slip Tickets per 100 employees',
        edgecolor='black')

ax2.bar(salary_pos, df_plot['SalaryPer100'], width=bar_width,
        color='lightgreen', label='Official Salary Slip Tickets per 100 employees',
        edgecolor='black')

# Trend lines
ax2.plot(df_plot['month_num'], df_plot['IncomePer100'],
         color='blue', linestyle='--', linewidth=1)
ax2.plot(df_plot['month_num'], df_plot['SalaryPer100'],
         color='green', linestyle='--', linewidth=1)

# Error bars
ax2.errorbar(income_pos,
             df_plot['IncomePer100'],
             yerr=df_plot['IncomeSTD100'],
             fmt='none', ecolor='blue', elinewidth=1.2, capsize=6)

ax2.errorbar(salary_pos,
             df_plot['SalaryPer100'],
             yerr=df_plot['SalarySTD100'],
             fmt='none', ecolor='green', elinewidth=1.2, capsize=6)

# Labels
ax2.set_title('Monthly Helpdesk Tickets per 100 Employees', fontsize=16)
ax2.set_xlabel('Month', fontsize=15)
ax2.set_ylabel('Tickets per 100 Employees', fontsize=15)
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax2.xaxis_date()
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=14)
plt.setp(ax2.get_yticklabels(), fontsize=14)
ax2.legend(fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(base_path, 'helpdesk_tickets_per100.png'),
             dpi=300, bbox_inches='tight', facecolor='white')

plt.show()