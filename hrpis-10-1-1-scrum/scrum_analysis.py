# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 19:59:39 2025

@author: Firas.Alhawari
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
folder_path = r"./"
input_file = os.path.join(folder_path, "scrum_input_filled.csv")
output_folder = os.path.join(folder_path, "scrum_outputs")
os.makedirs(output_folder, exist_ok=True)

sprint_weeks = 4
week_cols = [f"Week {i+1}" for i in range(sprint_weeks)]

# ----------------------------
# READ INPUT
# ----------------------------
df = pd.read_csv(input_file)

# Ensure defect columns exist
for col in ["Defects Found", "Defects Fixed", "Escaped Defects"]:
    df[col] = df.get(col, 0).fillna(0)

if "Bugs Fixed" not in df.columns:
    df["Bugs Fixed"] = 0
df["Bugs Fixed"] = df["Bugs Fixed"].fillna(0)

df[week_cols] = df[week_cols].fillna(0)
df['Done'] = False
df['Total Completion'] = 0

# ----------------------------
# SIMULATION SETUP
# ----------------------------
max_sprint = int(df['Planned Sprint'].max())
all_sprints = list(range(1, max_sprint + 1))

velocity_data = []
release_done = []
defect_data = []

running_outstanding = 0

# ----------------------------
# SIMULATE SPRINTS WEEK-BY-WEEK
# ----------------------------
for sprint in all_sprints:
    sprint_tasks = df[df['Planned Sprint'] == sprint]
    total_tasks_sprint = sprint_tasks.shape[0]
    remaining_per_week = []
    completed_this_sprint = 0

    for w in range(sprint_weeks):
        for idx, row in sprint_tasks.iterrows():
            if not df.at[idx, 'Done']:
                cumulative = row[week_cols[:w+1]].sum()
                if cumulative >= 100:
                    df.at[idx, 'Done'] = True
                    df.at[idx, 'Total Completion'] = 100
                    completed_this_sprint += 1
        remaining = df[(df['Planned Sprint'] == sprint) & (~df['Done'])].shape[0]
        remaining_per_week.append(remaining)

    velocity_data.append({
        'Sprint': sprint,
        'Completed Tasks': completed_this_sprint,
        'Not Completed': total_tasks_sprint - completed_this_sprint
    })

    # Defects calculation
    non_defect_tasks = sprint_tasks[sprint_tasks['Is Defect'] == 0]
    defects_found_sum = non_defect_tasks['Defects Found'].sum()
    defects_fixed_sum = non_defect_tasks['Defects Fixed'].sum()
    escaped_sum = non_defect_tasks['Escaped Defects'].sum()

    defect_tasks = sprint_tasks[sprint_tasks['Is Defect'] == 1]
    bugs_fixed_this_sprint = defect_tasks['Bugs Fixed'].sum()

    running_outstanding = escaped_sum + (defects_found_sum - defects_fixed_sum) - bugs_fixed_this_sprint
    running_outstanding = max(running_outstanding, 0)

    defect_data.append({
        "Sprint": sprint,
        "Defects Found": defects_found_sum,
        "Defects Fixed": defects_fixed_sum,
        "Escaped Defects": escaped_sum,
        "Bugs Fixed": bugs_fixed_this_sprint,
        "Total Outstanding": running_outstanding
    })

    # Sprint burndown
    plt.figure(figsize=(8,5))
    plt.plot(range(1, sprint_weeks+1), remaining_per_week, marker='o')
    plt.xticks(range(1, sprint_weeks+1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Week', fontsize=15)
    plt.ylabel('Remaining Tasks', fontsize=15)
    plt.title(f"Sprint {sprint} Burndown", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'sprint_{sprint}_burndown.png'))
    plt.show()

    release_done.append(df[df['Done']].shape[0])

# ----------------------------
# VELOCITY CHART
# ----------------------------
velocity_df = pd.DataFrame(velocity_data)
velocity_df.to_csv(os.path.join(output_folder, "velocity.csv"), index=False)

plt.figure(figsize=(8,5))
plt.bar(velocity_df['Sprint'], velocity_df['Completed Tasks'], label='Completed', alpha=0.7)
plt.bar(velocity_df['Sprint'], velocity_df['Not Completed'], bottom=velocity_df['Completed Tasks'], label='Not Completed', alpha=0.7)
plt.xlabel('Sprint', fontsize=15)
plt.ylabel('Tasks', fontsize=15)
plt.title('Sprint Velocity', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'velocity_chart_stacked.png'))
plt.show()

# ----------------------------
# DEFECTS CHART
# ----------------------------
defect_df = pd.DataFrame(defect_data)
defect_df.to_csv(os.path.join(output_folder, "defects_per_sprint.csv"), index=False)

plt.figure(figsize=(10,6))
plt.plot(defect_df['Sprint'], defect_df['Defects Found'], marker='o', color='red', label='Defects Found')
plt.plot(defect_df['Sprint'], defect_df['Defects Fixed'], marker='o', color='green', label='Defects Fixed')
plt.plot(defect_df['Sprint'], defect_df['Escaped Defects'], marker='o', color='orange', label='Escaped Defects')
plt.plot(defect_df['Sprint'], defect_df['Total Outstanding'], marker='o', linestyle='--', color='blue', label='Total Outstanding')
plt.plot(defect_df['Sprint'], defect_df['Bugs Fixed'], marker='s', linestyle='-', color='purple', label='Bugs Fixed (Defect Tasks)')
plt.xlabel('Sprint', fontsize=15)
plt.ylabel('Defects', fontsize=15)
plt.title('Defects Metrics per Sprint', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.ylim(0, 18)
ax = plt.gca()
ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "defects_line_chart.png"))
plt.show()

# ----------------------------
# RELEASE BURN-DOWN (CUMULATIVE) - SIMPLE FIX
# ----------------------------
cumulative_remaining = []

for sprint in all_sprints:
    # Tasks added up to this sprint
    tasks_added = df[df['Added Sprint'] <= sprint].shape[0]
    # Tasks planned in this sprint
    tasks_planned = df[df['Planned Sprint'] <= sprint].shape[0]
    # Remaining = added - planned (assume all planned are completed)
    remaining = tasks_added - tasks_planned
    cumulative_remaining.append(remaining)

plt.figure(figsize=(10,6))
plt.plot(all_sprints, cumulative_remaining, marker='o', color='blue', label='Remaining Tasks')
plt.xlabel('Sprint', fontsize=15)
plt.ylabel('Remaining Tasks', fontsize=15)
plt.title('Release Burn-Down', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.legend(fontsize=13, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'release_burndown.png'))
plt.show()

# ----------------------------
# RELEASE BURN-UP
# ----------------------------
release_df = pd.DataFrame({
    'Sprint': all_sprints,
    'Cumulative Completed': release_done,
    'Total Tasks': [df[df['Added Sprint'] <= s].shape[0] for s in all_sprints]
})
release_df.to_csv(os.path.join(output_folder, 'release_burnup.csv'), index=False)

plt.figure(figsize=(8,5))
plt.plot(release_df['Sprint'], release_df['Total Tasks'], linestyle='--', label='Total Tasks')
plt.plot(release_df['Sprint'], release_df['Cumulative Completed'], marker='o', label='Completed Tasks')
plt.xlabel('Sprint', fontsize=15)
plt.ylabel('Tasks', fontsize=15)
plt.title('Release Burn-Up', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'release_burnup.png'))
plt.show()

# ----------------------------
# SAVE BACKLOG
# ----------------------------
df.to_csv(os.path.join(output_folder, 'updated_backlog.csv'), index=False)
print("Scrum simulation completed. Outputs saved to:", output_folder)