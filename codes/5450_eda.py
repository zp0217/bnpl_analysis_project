import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid")

RED    = "red"
ORANGE = "orange"
BLUE   = "skyblue"
GRAY   = "gray"
DARK   = "black"



df = pd.read_csv("complaints.csv")
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(' ', '_').str.replace('?', '', regex=False))

df['date_received'] = pd.to_datetime(df['date_received'], format='%m/%d/%y', errors='coerce')
df['year']    = df['date_received'].dt.year
df['month']   = df['date_received'].dt.to_period('M')
df['quarter'] = df['date_received'].dt.to_period('Q')

df['got_relief']    = df['company_response_to_consumer'].str.contains('relief', case=False, na=False)
df['is_vulnerable'] = df['tags'].isin(['Servicemember', 'Older American', 'Older American, Servicemember'])
df['has_narrative'] = df['consumer_complaint_narrative'].notna()

product_map = {
    'Credit reporting or other personal consumer reports'            : 'Credit Reporting',
    'Credit reporting, credit repair services, or other personal consumer reports': 'Credit Reporting',
    'Debt collection'                                                : 'Debt Collection',
    'Payday loan, title loan, personal loan, or advance loan'        : 'Personal Loan/BNPL',
    'Payday loan, title loan, or personal loan'                      : 'Personal Loan/BNPL',
    'Credit card'                                                    : 'Credit Card',
    'Credit card or prepaid card'                                    : 'Credit Card',
    'Money transfer, virtual currency, or money service'             : 'Money Transfer',
    'Checking or savings account'                                    : 'Banking',
}
df['product_simple'] = df['product'].map(product_map).fillna('Other')


# fig 1

yearly = (df[df['year'].between(2023, 2025)]
          .groupby('year').size().reset_index(name='count'))

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = [GRAY, ORANGE, RED]
bars = ax.bar(yearly['year'].astype(str), yearly['count'],
              color=bar_colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, yearly['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
            f'{val:,}', ha='center', fontsize=12, fontweight='bold', color=DARK)



ax.set_title("Annual BNPL Complaint Volume (2023-2025)",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Complaints")
ax.set_ylim(0, yearly['count'].max() * 1.3)
plt.tight_layout()
plt.savefig("eda_fig1.png")
plt.show()



# fig 2 

top10 = df['issue'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [RED]*3 + [ORANGE]*3 + [BLUE]*4
bars = ax.barh(top10.index[::-1], top10.values[::-1],
               color=bar_colors[::-1], edgecolor='white')

for bar, val in zip(bars, top10.values[::-1]):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)

ax.set_title("Top 10 Complaint Issues in BNPL Services",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Number of Complaints")
ax.tick_params(axis='y', labelsize=9)

patches = [mpatches.Patch(color=RED,    label='Top 3'),
           mpatches.Patch(color=ORANGE, label='Rank 4–6'),
           mpatches.Patch(color=BLUE,   label='Rank 7–10')]
ax.legend(handles=patches, fontsize=9)
plt.tight_layout()
plt.savefig("eda_fig2.png", bbox_inches='tight')
plt.show()



#fig 3

issue_relief = (df.groupby('issue')
                  .agg(total=('got_relief','count'), relief=('got_relief','sum'))
                  .assign(relief_rate=lambda x: x['relief']/x['total']*100)
                  .query('total >= 30')
                  .sort_values('total', ascending=False)
                  .head(12))

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [RED if r < 1 else ORANGE if r < 3 else BLUE
              for r in issue_relief['relief_rate']]
bars = ax.barh(issue_relief.index[::-1],
               issue_relief['relief_rate'].values[::-1],
               color=bar_colors[::-1], edgecolor='white')

for bar, (rate, total) in zip(bars,
        zip(issue_relief['relief_rate'].values[::-1],
            issue_relief['total'].values[::-1])):
    label = f'{rate:.1f}%   (n={total:,})'
    ax.text(max(bar.get_width(), 0) + 0.05,
            bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=8.5)

avg = df['got_relief'].mean() * 100
ax.axvline(avg, color=DARK, linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(avg + 0.05, len(issue_relief) - 0.3,
        f'Overall avg: {avg:.1f}%', color=DARK, fontsize=8)

ax.set_title("Relief Rate by Complaint Issue",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Relief Rate (%)")
ax.tick_params(axis='y', labelsize=8.5)
patches = [mpatches.Patch(color=RED,    label='< 1% relief'),
           mpatches.Patch(color=ORANGE, label='1–3% relief'),
           mpatches.Patch(color=BLUE,   label='> 3% relief')]
ax.legend(handles=patches, fontsize=9)
plt.tight_layout()
plt.savefig("eda_fig3.png", bbox_inches='tight')
plt.show()

# fig 4

resp_counts = df['company_response_to_consumer'].value_counts()
total = len(df)

fig, ax = plt.subplots(figsize=(8, 3.5))
resp_colors = [RED if 'explanation' in r.lower() else
               BLUE if 'relief' in r.lower() else GRAY
               for r in resp_counts.index]
bars = ax.barh(resp_counts.index[::-1],
               (resp_counts.values / total * 100)[::-1],
               color=resp_colors[::-1], edgecolor='white', height=0.45)

for bar, (cnt, pct) in zip(bars,
        zip(resp_counts.values[::-1],
            (resp_counts.values / total * 100)[::-1])):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%  ({cnt:,} cases)', va='center', fontsize=10)

ax.set_title("Bar Graph: Outcome of BNPL Consumer Complaints",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("% of All Complaints")
ax.set_xlim(0, 120)
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
plt.savefig("eda_fig4.png", bbox_inches='tight')
plt.show()



#fig 5

COLORS = {
    'High complaints + High relief': 'orange',   # orange
    'High complaints + Low relief':  'red',   # red
    'Low complaints + High relief':  'green',   # green
    'Low complaints + Low relief':   'skyblue',   # blue
}

GRAY = 'gray'
DARK = 'black'


prod_relief = (
    df.groupby('product_simple')
      .agg(
          total=('got_relief', 'count'),
          relief=('got_relief', 'sum')
      )
      .assign(relief_rate=lambda x: x['relief'] / x['total'] * 100)
      .sort_values('total', ascending=False)
)

x = prod_relief['total'].values
y = prod_relief['relief_rate'].values
labels = prod_relief.index.tolist()

x_mid = np.median(x)
y_mid = np.median(y)

def classify_quadrant(xi, yi, x_mid, y_mid):
    if xi >= x_mid and yi >= y_mid:
        return 'High complaints + High relief'
    elif xi >= x_mid and yi < y_mid:
        return 'High complaints + Low relief'
    elif xi < x_mid and yi >= y_mid:
        return 'Low complaints + High relief'
    else:
        return 'Low complaints + Low relief'

quadrants = [classify_quadrant(xi, yi, x_mid, y_mid) for xi, yi in zip(x, y)]


fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

x_pad = max(300, x.max() * 0.07)
y_pad = max(0.25, y.max() * 0.08)

x_left, x_right = 0, x.max() + x_pad
y_bottom, y_top = 0, y.max() + y_pad

ax.set_xlim(x_left, x_right)
ax.set_ylim(y_bottom, y_top)

ax.axvline(x_mid, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.75, zorder=1)
ax.axhline(y_mid, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.75, zorder=1)

label_style = {
    'Other':               dict(xytext=(8, 8),    ha='left',  va='bottom'),
    'Money Transfer':      dict(xytext=(8, 4),    ha='left',  va='bottom'),
    'Credit Card':         dict(xytext=(8, 4),    ha='left',  va='bottom'),
    'Banking':             dict(xytext=(8, -8),   ha='left',  va='top'),
    'Personal Loan/BNPL':  dict(xytext=(-8, 10),  ha='right', va='bottom'),
    'Debt Collection':     dict(xytext=(8, 10),   ha='left',  va='bottom'),
    'Credit Reporting':    dict(xytext=(-10, 4),  ha='right', va='bottom'),
}

for xi, yi, label, quad in zip(x, y, labels, quadrants):
    color = COLORS[quad]

    ax.scatter(
        xi, yi,
        s=190,
        color=color,
        zorder=5,
        edgecolors='white',
        linewidth=1.6
    )

    spec = label_style.get(label, dict(xytext=(8, 4), ha='left', va='bottom'))

    ax.annotate(
        label,
        (xi, yi),
        textcoords='offset points',
        xytext=spec['xytext'],
        fontsize=9,
        color=DARK,
        fontweight='bold' if quad == 'High complaints + Low relief' else 'normal',
        ha=spec['ha'],
        va=spec['va'],
        clip_on=False
    )

left_x   = x_left + (x_mid - x_left) * 0.48
right_x  = x_mid + (x_right - x_mid) * 0.62
top_y    = y_mid + (y_top - y_mid) * 0.72
bottom_y = y_bottom + (y_mid - y_bottom) * 0.68

ax.text(
    left_x,
    top_y,
    'Low complaints\nHigh relief',
    fontsize=9,
    color=COLORS['Low complaints + High relief'],
    alpha=0.9,
    ha='center',
    va='center',
    fontweight='bold'
)

ax.text(
    right_x,
    top_y,
    'High complaints\nHigh relief',
    fontsize=9,
    color=COLORS['High complaints + High relief'],
    alpha=0.9,
    ha='center',
    va='center',
    fontweight='bold'
)

ax.text(
    left_x,
    bottom_y,
    'Low complaints\nLow relief',
    fontsize=9,
    color=COLORS['Low complaints + Low relief'],
    alpha=0.9,
    ha='center',
    va='center',
    fontweight='bold'
)

ax.text(
    right_x,
    bottom_y,
    'High complaints\nLow relief',
    fontsize=9,
    color=COLORS['High complaints + Low relief'],
    alpha=0.95,
    ha='center',
    va='center',
    fontweight='bold'
)

legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='High complaints + High relief',
        markerfacecolor=COLORS['High complaints + High relief'],
        markeredgecolor='white',
        markeredgewidth=1.5,
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='High complaints + Low relief',
        markerfacecolor=COLORS['High complaints + Low relief'],
        markeredgecolor='white',
        markeredgewidth=1.5,
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='Low complaints + High relief',
        markerfacecolor=COLORS['Low complaints + High relief'],
        markeredgecolor='white',
        markeredgewidth=1.5,
        markersize=9
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='Low complaints + Low relief',
        markerfacecolor=COLORS['Low complaints + Low relief'],
        markeredgecolor='white',
        markeredgewidth=1.5,
        markersize=9
    ),
]

ax.legend(
    handles=legend_elements,
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor="lightgray",
    fontsize=9,
    title='Quadrant'
)


ax.grid(True, color="silver", linewidth=0.8, alpha=0.75)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_color("lightgray")
    ax.spines[spine].set_linewidth(1.2)

ax.tick_params(colors=DARK)
ax.set_xlabel("Number of Complaints", fontsize=13, color=DARK)
ax.set_ylabel("Relief Rate (%)", fontsize=13, color=DARK)

ax.set_title(
    "Complaint Volume vs. Relief Rate by Product Category",
    fontsize=12,
    fontweight='bold',
    color=DARK,
    pad=14
)

plt.tight_layout(pad=1.3)
plt.savefig("eda_fig5", bbox_inches='tight', dpi=200)
plt.show()

