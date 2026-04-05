# ==========================================================
# analysis.py -- Pre & Post-DSEK Eye Health Tracker
# Mohammad Idrees  | GCET Kashmir | midrees-ai
# ==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ------ STEP1: Load the dataset ------
df = pd.read_csv(r'c:\users\moham\onedrive\desktop\dsek-eye-health-tracker\eye_data.csv')

# ------ STEP 2: Show basic info ------
print("Dataset loaded successfully!")
print(f"Total visits: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\n--- Basic Statistics ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum())


# ------ CHART 1: IOP Over Time ------
df_pre = df[(df["phase"] == "pre") & (df["iop_left"] > 0)]
plt.figure(figsize=(12, 5))
plt.plot(df_pre["days_from_surgery"], df_pre["iop_left"],
         marker="o", color="steelblue", linewidth=2, markersize=7)
plt.axhline(y=18, color="red", linestyle="--", label="High Risk Threshold (IOP=18)")
plt.axhline(y=12, color="green", linestyle="--", label="Normal Lower Bound (IOP=12)")
plt.title("IOP Left Eye Over Time (Pre-Surgery Visits)")
plt.xlabel("Days from Surgery (negative = before surgery)")
plt.ylabel("IOP (mmHg)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart1_iop_over_time.png', dpi=150)
print('Chart 1 saved!')
plt.close()



# ------ CHART 2: IOP by Hospital ------
df_iop = df[(df["iop_left"] > 0)]
plt.figure(figsize=(8, 5))
colors_hosp = {"sharp_sight": "steelblue", "dr_agarwals": "coral", "perfect_vision": "green"}
for hospital, group in df_iop.groupby("hospital"):
    plt.scatter(group["days_from_surgery"], group["iop_left"],
                label=hospital, s=80, color=colors_hosp.get(hospital, "gray"))
plt.axhline(y=18, color="red", linestyle="--", alpha=0.7, label="High Risk = 18")
plt.title("IOP Readings by Hospital")
plt.xlabel("Days from Surgery")
plt.ylabel("IOP Left Eye")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart2_iop_by_hospital.png', dpi=150)
print('Chart 2 saved!')
plt.close()

# ------ CHART 3: Recovery Stages ------
df_post = df[df["phase"] == "post"]
stage_colors = {"early": "#ef5350", "mid": "#ff9800", "good": "#4caf50"}
plt.figure(figsize=(10, 5))
for stage, group in df_post.groupby("recovery_stage"):
    plt.scatter(group["days_from_surgery"], [stage]*len(group),
                color=stage_colors.get(stage, "blue"), s=150, label=stage)
plt.title("Recovery Stage Progression After DSAEK Surgery")
plt.xlabel("Days After Surgery")
plt.ylabel("Recovery Stage")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart3_recovery_stages.png', dpi=150)
print('Chart 3 saved!')
plt.close()


# ------ CHART 4: Medicines Per Visit ------
df_pre2 = df[df["phase"] == "pre"]
plt.figure(figsize=(10, 5))
plt.bar(range(len(df_pre2)), df_pre2["num_medicines"],
        color=["red" if iop >= 18 else "steelblue" for iop in df_pre2["iop_left"]])
plt.title("Number of Medicines Prescribed Per Visit (Pre-Surgery)")
plt.xlabel("Visit Number")
plt.ylabel("Number of Medicines")
red_patch = mpatches.Patch(color="red", label="High IOP Visit")
blue_patch = mpatches.Patch(color="steelblue", label="Normal IOP Visit")
plt.legend(handles=[red_patch, blue_patch])
plt.tight_layout()
plt.savefig('chart4_medicines_per_visit.png', dpi=150)
print('Chart 4 saved!')
plt.close()


# ------ CHART 5: Correlation Heatmap ------
df_numeric = df.select_dtypes(include="number")
plt.figure(figsize=(10, 7))
sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Heatmap — Eye Health Features")
plt.tight_layout()
plt.savefig('chart5_correlation_heatmap.png', dpi=150)
print('Chart 5 saved!')
plt.close()

print('\nAll Day 1 charts saved successfully!')
