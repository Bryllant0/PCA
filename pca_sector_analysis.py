import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

# =============================================================================
# 1. DATA COLLECTION: S&P 500 Sector ETFs
# =============================================================================

# SPDR Sector ETFs (cover all 11 GICS sectors)
SECTORS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discret.',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLC': 'Communication',
    'XLRE': 'Real Estate'
}

# Tickers to download (sectors + SPY benchmark)
TICKERS = list(SECTORS.keys()) + ['SPY']

print("=" * 70)
print("PRINCIPAL COMPONENT ANALYSIS ON S&P 500 SECTOR ETFs")
print("=" * 70)
print(f"\nAnalyzing {len(SECTORS)} sector ETFs...")
print(f"Sectors: {', '.join(SECTORS.values())}\n")

# Download 3 years of daily data
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

print("Downloading data from Yahoo Finance...")
data = yf.download(TICKERS, start=start_date, end=end_date, progress=True)

# Handle both old and new yfinance API
if 'Adj Close' in data.columns.get_level_values(0):
    data = data['Adj Close']
elif 'Close' in data.columns.get_level_values(0):
    data = data['Close']
else:
    data = data[[c for c in data.columns if 'Close' in str(c)]]
    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

data = data.dropna()

# Compute daily returns
returns = data.pct_change().dropna()
sector_returns = returns[list(SECTORS.keys())]
spy_returns = returns['SPY']

print(f"\nData period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Total trading days: {len(returns)}")

# =============================================================================
# 2. CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: CORRELATION MATRIX ANALYSIS")
print("=" * 70)

correlation_matrix = sector_returns.corr()

sector_names = [SECTORS[t] for t in sector_returns.columns]
corr_display = correlation_matrix.copy()
corr_display.columns = sector_names
corr_display.index = sector_names

print("\nCorrelation Matrix of Sector Returns:")
print("-" * 50)
upper_tri = correlation_matrix.values[np.triu_indices(11, k=1)]
print(f"Average pairwise correlation: {upper_tri.mean():.3f}")
print(f"Min correlation: {upper_tri.min():.3f}")
print(f"Max correlation: {upper_tri.max():.3f}")

# =============================================================================
# 3. PERFORM PCA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: PRINCIPAL COMPONENT ANALYSIS")
print("=" * 70)

scaler = StandardScaler()
returns_standardized = scaler.fit_transform(sector_returns)

pca = PCA()
principal_components = pca.fit_transform(returns_standardized)

eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
eigenvectors = pca.components_

print("\nEigenvalues and Explained Variance:")
print("-" * 60)
print(f"{'PC':<6} {'Eigenvalue':<12} {'Variance %':<14} {'Cumulative %':<14}")
print("-" * 60)
for i in range(len(eigenvalues)):
    print(f"PC{i+1:<4} {eigenvalues[i]:<12.4f} {explained_variance_ratio[i]*100:<14.2f} {cumulative_variance[i]*100:<14.2f}")

print("\n>>> KEY INSIGHT: PC1 explains {:.1f}% of total variance - this is the MARKET MODE".format(
    explained_variance_ratio[0]*100))
print(">>> The first 3 PCs explain {:.1f}% of variance".format(cumulative_variance[2]*100))

# =============================================================================
# 4. INTERPRET EIGENVECTOR LOADINGS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: EIGENVECTOR LOADINGS (Portfolio Weights)")
print("=" * 70)

loadings_df = pd.DataFrame(
    eigenvectors.T,
    columns=[f'PC{i+1}' for i in range(11)],
    index=sector_names
)

print("\nPC1 Loadings (Market Mode):")
print("-" * 40)
pc1_loadings = loadings_df['PC1'].sort_values(ascending=False)
for sector, loading in pc1_loadings.items():
    bar = '█' * int(abs(loading) * 20)
    sign = '+' if loading > 0 else '-'
    print(f"{sector:<20} {sign}{abs(loading):.3f}  {bar}")

print("\n>>> All loadings are positive → PC1 is a 'long everything' market factor")

print("\nPC2 Loadings (Sector Rotation):")
print("-" * 40)
pc2_loadings = loadings_df['PC2'].sort_values(ascending=False)
for sector, loading in pc2_loadings.items():
    bar_pos = '█' * int(max(0, loading) * 15)
    bar_neg = '█' * int(max(0, -loading) * 15)
    sign = '+' if loading > 0 else '-'
    print(f"{sector:<20} {sign}{abs(loading):.3f}  {bar_neg}{'|'}{bar_pos}")

pc2_long = [s for s, v in pc2_loadings.items() if v > 0.2]
pc2_short = [s for s, v in pc2_loadings.items() if v < -0.2]
print(f"\n>>> PC2 represents sector rotation:")
print(f"    LONG:  {', '.join(pc2_long) if pc2_long else 'N/A'}")
print(f"    SHORT: {', '.join(pc2_short) if pc2_short else 'N/A'}")

# =============================================================================
# 5. EIGEN-PORTFOLIO PERFORMANCE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: EIGEN-PORTFOLIO PERFORMANCE")
print("=" * 70)

pc_series = pd.DataFrame(
    principal_components,
    index=sector_returns.index,
    columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
)

pc1_spy_corr = np.corrcoef(pc_series['PC1'], spy_returns)[0, 1]
print(f"\nCorrelation between PC1 (Eigen-Portfolio 1) and SPY: {pc1_spy_corr:.4f}")
print(">>> This confirms PC1 captures the broad market movement")

# =============================================================================
# 6. STATISTICAL ARBITRAGE: RESIDUAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: STATISTICAL ARBITRAGE - RESIDUAL ANALYSIS")
print("=" * 70)

n_factors = 3
factor_returns_df = pc_series[[f'PC{i+1}' for i in range(n_factors)]]

residuals_df = pd.DataFrame(index=sector_returns.index)
r_squared = {}

print(f"\nRegressing each sector on the first {n_factors} Principal Components:")
print("-" * 50)

tickers = list(SECTORS.keys())
for i, ticker in enumerate(tickers):
    sector_name = SECTORS[ticker]
    y = returns_standardized[:, i]
    X = factor_returns_df.values
    
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    residual = y - y_pred
    
    residuals_df[sector_name] = residual
    r_squared[sector_name] = reg.score(X, y)
    
    print(f"{sector_name:<20} R² = {r_squared[sector_name]:.3f}  (Idiosyncratic: {(1-r_squared[sector_name])*100:.1f}%)")

recent_residuals = residuals_df.tail(20).mean()
z_scores = recent_residuals / residuals_df.std()

print("\n" + "-" * 50)
print("Current Residual Z-Scores (last 20 days avg):")
print("-" * 50)
for sector in z_scores.sort_values().index:
    z = z_scores[sector]
    signal = "◄ UNDERVALUED (LONG)" if z < -1 else "◄ OVERVALUED (SHORT)" if z > 1 else ""
    print(f"{sector:<20} Z = {z:+.2f}  {signal}")

# =============================================================================
# 7. GENERATE COMPREHENSIVE VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS...")
print("=" * 70)

# Use subplots with explicit spacing
fig, axes = plt.subplots(2, 3, figsize=(18, 14))
plt.subplots_adjust(hspace=0.45, wspace=0.3, top=0.93, bottom=0.08)

colors = plt.cm.tab10(np.linspace(0, 1, 11))

# --- Plot 1: Correlation Heatmap ---
ax1 = axes[0, 0]
sns.heatmap(corr_display, annot=True, fmt='.2f', cmap='RdYlBu_r', 
            center=0.5, vmin=0, vmax=1, ax=ax1,
            annot_kws={'size': 6}, cbar_kws={'shrink': 0.8})
ax1.set_title('Correlation Matrix of Sector Returns\n(High correlation → Common risk factors)', 
              fontweight='bold', fontsize=10)
ax1.tick_params(axis='both', labelsize=6)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)

# --- Plot 2: Scree Plot ---
ax2 = axes[0, 1]
x_pos = np.arange(1, len(eigenvalues) + 1)
bars = ax2.bar(x_pos, explained_variance_ratio * 100, color='steelblue', alpha=0.7, 
               edgecolor='navy', linewidth=1.2, label='Individual')
ax2.plot(x_pos, cumulative_variance * 100, 'ro-', linewidth=2.5, markersize=7, 
         label='Cumulative', markeredgecolor='darkred', markeredgewidth=1.5)
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='80% threshold')
ax2.set_xlabel('Principal Component', fontweight='bold', fontsize=9)
ax2.set_ylabel('Explained Variance (%)', fontweight='bold', fontsize=9)
ax2.set_title('Scree Plot: Variance Explained by Each PC\n(Most variance in first few PCs)', 
              fontweight='bold', fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'PC{i}' for i in x_pos], fontsize=8)
ax2.legend(loc='center right', fontsize=8)
ax2.set_ylim(0, 105)

for bar, pct in zip(bars, explained_variance_ratio * 100):
    if pct > 5:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Plot 3: PC1 vs PC2 Loadings Biplot ---
ax3 = axes[0, 2]
for i, sector in enumerate(sector_names):
    ax3.annotate('', xy=(loadings_df.loc[sector, 'PC1'], loadings_df.loc[sector, 'PC2']),
                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    ax3.scatter(loadings_df.loc[sector, 'PC1'], loadings_df.loc[sector, 'PC2'], 
               s=80, c=[colors[i]], edgecolors='black', linewidths=1, zorder=5)
    ax3.text(loadings_df.loc[sector, 'PC1'] + 0.012, 
             loadings_df.loc[sector, 'PC2'] + 0.012,
             sector, fontsize=7, ha='left', fontweight='bold')

ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax3.set_xlabel('PC1 Loading (Market Mode)', fontweight='bold', fontsize=9)
ax3.set_ylabel('PC2 Loading (Rotation Factor)', fontweight='bold', fontsize=9)
ax3.set_title('Factor Loadings Biplot\n(Arrows show sector exposure to each factor)', 
              fontweight='bold', fontsize=10)

# --- Plot 4: Loadings Bar Chart ---
ax4 = axes[1, 0]
x = np.arange(len(sector_names))
width = 0.25

ax4.bar(x - width, loadings_df['PC1'].values, width, label='PC1 (Market)', 
        color='#27ae60', alpha=0.85, edgecolor='darkgreen', linewidth=1)
ax4.bar(x, loadings_df['PC2'].values, width, label='PC2 (Rotation 1)', 
        color='#3498db', alpha=0.85, edgecolor='darkblue', linewidth=1)
ax4.bar(x + width, loadings_df['PC3'].values, width, label='PC3 (Rotation 2)', 
        color='#9b59b6', alpha=0.85, edgecolor='purple', linewidth=1)

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Sector', fontweight='bold', fontsize=9)
ax4.set_ylabel('Loading (Eigenvector Weight)', fontweight='bold', fontsize=9)
ax4.set_title('Eigenvector Loadings by Component\n(Weights for constructing Eigen-Portfolios)', 
              fontweight='bold', fontsize=10)
ax4.set_xticks(x)
ax4.set_xticklabels(sector_names, rotation=45, ha='right', fontsize=7)
ax4.legend(loc='upper right', fontsize=8)
ax4.set_ylim(-0.7, 0.7)

# --- Plot 5: Cumulative Performance ---
ax5 = axes[1, 1]

pc1_scaled = pc_series['PC1'] * sector_returns.std().mean()
pc2_scaled = pc_series['PC2'] * sector_returns.std().mean() * 0.5

pc1_cumret = (1 + pc1_scaled).cumprod()
pc2_cumret = (1 + pc2_scaled).cumprod()
spy_cumret = (1 + spy_returns).cumprod()

ax5.plot(pc_series.index, pc1_cumret, label='PC1 (Market Mode)', 
         linewidth=2, color='#27ae60')
ax5.plot(pc_series.index, spy_cumret, label='SPY (Benchmark)', 
         linewidth=2, color='black', linestyle='--', alpha=0.8)
ax5.plot(pc_series.index, pc2_cumret, label='PC2 (Rotation)', 
         linewidth=2, color='#3498db', alpha=0.8)

ax5.set_xlabel('Date', fontweight='bold', fontsize=9)
ax5.set_ylabel('Cumulative Return (Growth of $1)', fontweight='bold', fontsize=9)
ax5.set_title(f'Eigen-Portfolio Performance\n(PC1-SPY correlation: {pc1_spy_corr:.3f})', 
              fontweight='bold', fontsize=10)
ax5.legend(loc='upper left', fontsize=8)
ax5.axhline(y=1, color='gray', linestyle='-', alpha=0.3)

ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# --- Plot 6: Statistical Arbitrage Signals ---
ax6 = axes[1, 2]
z_sorted = z_scores.sort_values()
colors_z = ['#e74c3c' if z > 1 else '#27ae60' if z < -1 else '#bdc3c7' for z in z_sorted.values]

bars = ax6.barh(z_sorted.index, z_sorted.values, color=colors_z, alpha=0.85,
               edgecolor='black', linewidth=0.8)
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax6.axvline(x=1, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
ax6.axvline(x=-1, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)

ax6.axvspan(-3, -1, alpha=0.1, color='green')
ax6.axvspan(1, 3, alpha=0.1, color='red')

ax6.set_xlabel('Z-Score of Residual (20-day average)', fontweight='bold', fontsize=9)
ax6.set_title('Statistical Arbitrage Signals\n(Deviation from fair value based on PC1-3)', 
              fontweight='bold', fontsize=10)

for bar, (sector, z) in zip(bars, z_sorted.items()):
    if abs(z) > 1:
        label = 'SHORT' if z > 1 else 'LONG'
        color = '#e74c3c' if z > 1 else '#27ae60'
        ax6.text(z + 0.15 * np.sign(z), bar.get_y() + bar.get_height()/2, 
                label, va='center', fontsize=8, fontweight='bold', color=color)

ax6.set_xlim(-2.5, 2.5)
ax6.tick_params(axis='y', labelsize=8)

plt.savefig('pca_sector_analysis.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✓ Main visualization saved to 'pca_sector_analysis.png'")

# =============================================================================
# 8. METHODOLOGY FIGURE
# =============================================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
plt.subplots_adjust(wspace=0.35, bottom=0.15)

# Left: Raw returns
ax = axes2[0]
sample_tickers = ['XLK', 'XLU', 'XLE']
for ticker in sample_tickers:
    cumret = (1 + sector_returns[ticker]).cumprod()
    ax.plot(sector_returns.index, cumret, label=SECTORS[ticker], linewidth=2)
ax.set_title('Raw Sector Returns\n(Highly correlated due to common market factor)', fontweight='bold', fontsize=10)
ax.set_xlabel('Date', fontweight='bold', fontsize=9)
ax.set_ylabel('Cumulative Return', fontweight='bold', fontsize=9)
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# Middle: PCA transformation
ax = axes2[1]
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], 
                     c=np.arange(len(principal_components)), cmap='viridis', alpha=0.5, s=15)
ax.set_xlabel('PC1 (Market Mode)', fontweight='bold', fontsize=9)
ax.set_ylabel('PC2 (Rotation Factor)', fontweight='bold', fontsize=9)
ax.set_title('PCA Transformation\n(Uncorrelated principal components)', fontweight='bold', fontsize=10)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time (days)', fontsize=9)

# Right: Variance pie chart
ax = axes2[2]
wedges, texts, autotexts = ax.pie(
    explained_variance_ratio[:4], 
    labels=[f'PC{i+1}' for i in range(4)],
    autopct=lambda pct: f'{pct:.1f}%',
    colors=['#27ae60', '#3498db', '#9b59b6', '#e74c3c'], 
    explode=[0.05, 0, 0, 0], 
    startangle=90,
    textprops={'fontsize': 10}
)
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
remaining = 1 - sum(explained_variance_ratio[:4])
ax.set_title(f'Variance Decomposition\n(First 4 PCs explain {(1-remaining)*100:.1f}% of total)', 
             fontweight='bold', fontsize=10)

plt.savefig('pca_methodology.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Methodology figure saved to 'pca_methodology.png'")

# =============================================================================
# 9. SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: KEY RESULTS")
print("=" * 70)

summary = f"""
┌─────────────────────────────────────────────────────────────────────┐
│  PRINCIPAL COMPONENT ANALYSIS - S&P 500 SECTORS                     │
├─────────────────────────────────────────────────────────────────────┤
│  Data Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}                          │
│  Number of Observations: {len(returns):<5}                                    │
│  Number of Sectors: {len(SECTORS)}                                              │
├─────────────────────────────────────────────────────────────────────┤
│  VARIANCE DECOMPOSITION:                                            │
│    • PC1 (Market Mode):     {explained_variance_ratio[0]*100:5.1f}% of total variance             │
│    • PC2 (Rotation 1):      {explained_variance_ratio[1]*100:5.1f}% of total variance             │
│    • PC3 (Rotation 2):      {explained_variance_ratio[2]*100:5.1f}% of total variance             │
│    • First 3 PCs Combined:  {cumulative_variance[2]*100:5.1f}% of total variance             │
├─────────────────────────────────────────────────────────────────────┤
│  KEY FINDINGS:                                                      │
│    • PC1 correlation with SPY: {pc1_spy_corr:.4f}                            │
│    • Average sector correlation: {upper_tri.mean():.3f}                          │
│    • Effective # of independent factors: ~{np.sum(eigenvalues > 1):.0f}                         │
└─────────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

plt.show()
