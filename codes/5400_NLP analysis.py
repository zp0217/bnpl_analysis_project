# NLP analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid")

RED    = "red"
ORANGE = "orange"
BLUE   = "skyblue"
GRAY   = "gray"
DARK   = "black"


# load, clean data

df = pd.read_csv("complaints.csv")
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(' ', '_').str.replace('?', '', regex=False))
df['got_relief'] = df['company_response_to_consumer'].str.contains(
    'relief', case=False, na=False)

STOP = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn','make','made','go','going','get','got','one','two','would',
    'also','even','said','told','called','us','since','know','back','need',
    'want','think','still','never','always','many','much','take','see','say',
    'company','affirm','klarna','account','payment','loan','purchase','use',
    'per','new','first','last','day','time','month','year','like','well',
}

def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) < 30:
        return None
    text = text.lower()
    text = re.sub(r'\bx+\b', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP and len(t) > 2]
    return ' '.join(tokens) if len(tokens) >= 8 else None

nlp_df = df[df['consumer_complaint_narrative'].notna()].copy()
nlp_df['cleaned'] = nlp_df['consumer_complaint_narrative'].apply(clean_text)
nlp_df = nlp_df[nlp_df['cleaned'].notna()].reset_index(drop=True)

all_texts     = nlp_df['cleaned'].tolist()
relief_texts  = nlp_df[nlp_df['got_relief']]['cleaned'].tolist()
norelief_texts= nlp_df[~nlp_df['got_relief']]['cleaned'].tolist()


# fig 1

combined = ' '.join(all_texts)
wc = WordCloud(
    width=1200, height=500,
    background_color='white',
    colormap='RdYlBu_r',
    max_words=150,
    collocations=False,
    prefer_horizontal=0.75
).generate(combined)

fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title("Word Cloud: Consumer Complaint Narratives",
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("nlp_fig1", bbox_inches='tight')
plt.show()





#fig 2

def get_top_tfidf(texts, n=15):
    if len(texts) < 5:
        return pd.Series(dtype=float)
    v = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X = v.fit_transform(texts)
    s = np.asarray(X.mean(axis=0)).flatten()
    w = v.get_feature_names_out()
    idx = s.argsort()[::-1][:n]
    return pd.Series(s[idx], index=w[idx])

relief_kw   = get_top_tfidf(relief_texts,   15)
norelief_kw = get_top_tfidf(norelief_texts, 15)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (kw, label, color, subtitle) in zip(axes, [
        (relief_kw,   'Got Relief',     BLUE,   f'n={len(relief_texts):,} complaints'),
        (norelief_kw, 'No Relief',      RED,    f'n={len(norelief_texts):,} complaints')]):
    s = kw.sort_values()
    ax.barh(s.index, s.values, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f"{label}\n({subtitle})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Mean TF-IDF Score")
    ax.tick_params(axis='y', labelsize=9)

plt.suptitle("Keyword Comparison: Resolved vs Unresolved Complaints",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("nlp_fig2.png", bbox_inches='tight')
plt.show()



#fig 3

reg_keywords = {
    'FCRA / Credit Rights' : ['fcra', 'fair credit', 'credit reporting act'],
    'FDCPA / Debt Rights'  : ['fdcpa', 'fair debt', 'debt collection act'],
    'Dispute Rights'       : ['dispute', 'chargeback', 'right to dispute'],
    'Disclosure Issues'    : ['disclosure', 'not disclosed', 'not informed', 'hidden fee'],
    'Fraud / Identity Theft': ['fraud', 'identity theft', 'unauthorized'],
    'TILA-related'         : ['interest rate', 'apr', 'annual percentage', 'finance charge'],
}

narratives_raw = nlp_df['consumer_complaint_narrative'].str.lower().fillna('')
freq_data = {}
for category, terms in reg_keywords.items():
    pattern = '|'.join(terms)
    count   = narratives_raw.str.contains(pattern, na=False).sum()
    pct     = count / len(narratives_raw) * 100
    freq_data[category] = {'count': count, 'pct': pct}

freq_df = pd.DataFrame(freq_data).T.sort_values('pct', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = [RED if 'TILA' in i or 'Disclosure' in i else
              ORANGE if 'Fraud' in i else BLUE
              for i in freq_df.index]
bars = ax.barh(freq_df.index, freq_df['pct'], color=bar_colors, edgecolor='white')

for bar, (cnt, pct) in zip(bars, zip(freq_df['count'], freq_df['pct'])):
    ax.text(bar.get_width() + 0.2,
            bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%  ({cnt:,} cases)', va='center', fontsize=9)

ax.set_title("Regulatory Term Frequency in Consumer Narratives",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("% of All Narratives Containing This Term")
ax.set_xlim(0, freq_df['pct'].max() * 1.45)
ax.tick_params(axis='y', labelsize=9)

patches = [mpatches.Patch(color=RED,    label='TILA/Disclosure (regulatory gap)'),
           mpatches.Patch(color=ORANGE, label='Fraud/Unauthorized'),
           mpatches.Patch(color=BLUE,   label='Existing legal frameworks')]
ax.legend(handles=patches, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig("nlp_fig3.png", bbox_inches='tight')
plt.show()



#fig 4


vec_lda = CountVectorizer(max_features=3000, min_df=5,
                          ngram_range=(1,2), stop_words='english')
X_lda   = vec_lda.fit_transform(all_texts)
lda     = LatentDirichletAllocation(n_components=6, random_state=42,
                                    max_iter=30, learning_method='online')
lda.fit(X_lda)

feat_names = vec_lda.get_feature_names_out()
topics = {}
for i, comp in enumerate(lda.components_):
    top_words = [feat_names[j] for j in comp.argsort()[:-11:-1]]
    topics[i] = top_words


for i, words in topics.items():
    print(f"Topic {i+1}: {', '.join(words)}")

#topic table
TOPIC_LABELS = {
    0: "1. Credit Report\nInaccuracy",
    1: "2. FCRA /\nPrivacy Violation",
    2: "3. Refund &\nMerchant Dispute",
    3: "4. Debt\nCollection",
    4: "5. Unauthorized\nCharges & Fees",
    5: "6. Identity\nTheft / Fraud",
}

# fig 4
data_mat = {}
for i, words in topics.items():
    label = TOPIC_LABELS.get(i, f"T{i+1}")
    data_mat[label] = words[:8]

all_kw = list(dict.fromkeys(w for wl in data_mat.values() for w in wl))
matrix = pd.DataFrame(0.0, index=list(data_mat.keys()), columns=all_kw)
for topic, words in data_mat.items():
    for rank, word in enumerate(words):
        matrix.loc[topic, word] = len(words) - rank

fig, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(matrix, cmap='Reds', linewidths=0.4, ax=ax,
            cbar_kws={'label': 'Rank Score (higher = more central)'})
ax.set_title("LDA Topic Model — Topic-Keyword Heatmap",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Keywords")
ax.set_ylabel("Topics")
plt.xticks(rotation=35, ha='right', fontsize=8)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("nlp_fig4", bbox_inches='tight')
plt.show()



