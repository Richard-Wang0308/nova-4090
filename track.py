import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- 1. Connect to DB ----
DB_PATH = 'score_results_2.sqlite'
conn = sqlite3.connect(DB_PATH)

TABLE_NAME = 'scored_molecules'  # change if actual table name differs

query = f"SELECT scored_at, score FROM {TABLE_NAME}"
df = pd.read_sql_query(query, conn)
conn.close()

# ---- 2. Parse datetime & sort ----
df['scored_at'] = pd.to_datetime(df['scored_at'])
df = df.sort_values('scored_at')

# ---- 3. Group by timestamp: mean, max, and row count ----
grouped = (
    df.groupby('scored_at')['score']
      .agg(avg_score='mean', max_score='max', row_count='count')
      .reset_index()
      .sort_values('scored_at')
)

# ---- 4. Cumulative row count up to each timestamp ----
grouped['cumulative_rows'] = grouped['row_count'].cumsum()

print(f"Total unique timestamps: {len(grouped)}")
print(f"Total rows: {grouped['cumulative_rows'].iloc[-1]}")
print(grouped.head())

# ---- 5. Helper: pick a limited number of x-tick positions to avoid clutter ----
def get_tick_indices(n_points, max_ticks=10):
    if n_points <= max_ticks:
        return list(range(n_points))
    step = max(1, n_points // max_ticks)
    return list(range(0, n_points, step))

tick_idx = get_tick_indices(len(grouped))
tick_positions = grouped['cumulative_rows'].iloc[tick_idx]
tick_labels = [
    f"{cum}\n({t.strftime('%m-%d %H:%M:%S')})"
    for cum, t in zip(grouped['cumulative_rows'].iloc[tick_idx],
                       grouped['scored_at'].iloc[tick_idx])
]

# ---- 6. Plot: two subplots (Average / Max) ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# -- Average score --
ax1.plot(
    grouped['cumulative_rows'], grouped['avg_score'],
    marker='o', markersize=4, linewidth=1.5, color='#4CAF50'
)
ax1.set_title('Average Score per Timestamp', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Score')
ax1.grid(True, alpha=0.3)

# -- Max score --
ax2.plot(
    grouped['cumulative_rows'], grouped['max_score'],
    marker='o', markersize=4, linewidth=1.5, color='#FF5722'
)
ax2.set_title('Max Score per Timestamp', fontsize=13, fontweight='bold')
ax2.set_xlabel('Cumulative Rows Processed (Timestamp)')
ax2.set_ylabel('Max Score')
ax2.grid(True, alpha=0.3)

# ---- 7. Custom x-axis ticks (cumulative rows + time in brackets) ----
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('score_over_time_avg_max.png', dpi=150)
plt.show()
