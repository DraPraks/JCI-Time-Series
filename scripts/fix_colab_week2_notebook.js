const fs = require('fs');

const notebooks = [
  '01_EDA_cleaning.ipynb',
  'Week2_submission_JCI_Prediction.ipynb',
];

const featureNotebooks = [
  '02_feature_engineering_modeling.ipynb',
  'Week2_submission_JCI_Prediction.ipynb',
];

const sectorLoader = `# Load and combine by sector
# Use local IDX-IC sector CSV files when available. The embedded fallback keeps
# this notebook runnable in Colab when only the notebook itself was uploaded.

import glob

SECTOR_DIR = os.path.join('Dataset', 'stock-list-per-sector')
sector_files = glob.glob(os.path.join(SECTOR_DIR, '*.csv'))

sector_dfs = []
for f in sorted(sector_files):
    basename = os.path.basename(f)
    # filename format: "Stock List  - {Sector} - YYYYMMDD.csv"
    sector_name = basename.split(' - ')[1].strip()
    df_s = pd.read_csv(f)
    df_s['Sector'] = sector_name
    sector_dfs.append(df_s[['Code', 'Sector']])

if sector_dfs:
    sector_df = pd.concat(sector_dfs, ignore_index=True)
    sector_df = sector_df.rename(columns={'Code': 'Ticker'})
else:
    FALLBACK_SECTOR_TICKERS = {
        'Energy': [
            'ABMM', 'AKRA', 'APEX', 'ARII', 'ARTI', 'BBRM', 'BIPI', 'BSSR',
            'BULL', 'BUMI', 'BYAN', 'CANI', 'CNKO', 'DEWA', 'DOID', 'DSSA',
            'ELSA', 'ENRG', 'GEMS', 'GTBO', 'HITS', 'HRUM', 'IATA', 'INDY',
            'ITMA', 'ITMG', 'KKGI', 'KOPI', 'LEAD', 'MBAP', 'MBSS', 'MEDC',
            'MTFN', 'MYOH', 'PGAS', 'PKPK', 'PTBA', 'PTIS', 'PTRO', 'RAJA',
            'RIGS', 'RUIS', 'SMMT', 'SMRU', 'SOCI', 'SUGI', 'TOBA', 'TPMA',
            'TRAM', 'WINS', 'SHIP', 'TAMU', 'FIRE', 'PSSI', 'DWGL', 'BOSS',
            'JSKY', 'INPS', 'TCPI', 'SURE', 'WOWS', 'TEBE', 'SGER', 'UNIQ',
            'MCOL', 'GTSI', 'RMKE', 'BSML', 'ADMR', 'SEMA', 'SICO', 'COAL',
            'SUNI', 'CBRE', 'HILL', 'CUAN', 'MAHA', 'RMKO', 'HUMI', 'RGAS',
            'ALII', 'MKAP', 'ATLA', 'BOAT', 'AADI', 'RATU', 'PSAT', 'BESS',
            'CGAS', 'ADRO', 'AIMS',
        ],
        'Technology': [
            'ATIC', 'EMTK', 'KREN', 'LMAS', 'MLPT', 'MTDL', 'PTSN', 'SKYB',
            'KIOS', 'MCAS', 'NFCX', 'DIVA', 'LUCK', 'ENVY', 'HDIT', 'TFAS',
            'DMMX', 'GLVA', 'PGJO', 'CASH', 'TECH', 'EDGE', 'ZYRX', 'UVCR',
            'BUKA', 'RUNS', 'WGSH', 'WIRG', 'GOTO', 'AXIO', 'BELI', 'NINE',
            'ELIT', 'IRSX', 'CHIP', 'TRON', 'JATI', 'CYBR', 'IOTF', 'MSTI',
            'TOSK', 'MPIX', 'AREA', 'MENN', 'AWAN', 'WIFI', 'DCII',
        ],
    }
    sector_df = pd.DataFrame(
        [
            (ticker, sector)
            for sector, tickers in FALLBACK_SECTOR_TICKERS.items()
            for ticker in tickers
        ],
        columns=['Ticker', 'Sector'],
    )
    print('Sector CSV folder not found; using embedded Energy/Technology mapping.')

# Cross-reference with tickers present in the OHLCV dataset.
# unique_tickers is the ticker set created during the scale overview.
sector_df = sector_df[sector_df['Ticker'].isin(unique_tickers)].copy()

print(f"Sector files loaded:          {len(sector_files)}")
print(f"Total sector-ticker entries:  {len(sector_df):,}")
print(f"Unique tickers matched:       {sector_df['Ticker'].nunique():,}")
print(f"Unique sectors:               {sector_df['Sector'].nunique()}")
print(f"\\nTickers in OHLCV data but NOT in sector list: "
      f"{len(unique_tickers - set(sector_df['Ticker'])):,}")
`;

const exportSummary = `print(f"Tickers:           {df_model['Ticker'].nunique()}")
print(f"Total rows:        {len(df_model):,}")
print(f"Date range:        {df_model['Date'].min().date()} to {df_model['Date'].max().date()}")
print(f"Sectors:           {list(df_model['sector'].unique())}")
print("Sector distribution:")
print(df_model['sector'].value_counts())
print(f"Parquet files in {OUTPUT_DIR}:")
for f in sorted(_os.listdir(OUTPUT_DIR)):
    if f.endswith(".parquet"):
        fp = os.path.join(OUTPUT_DIR, f)
        mb = _os.path.getsize(fp) / (1024**2)
        print(f"  {f:<30} {mb:>6.1f} MB")
print(f"Rows per ticker: min={df_model.groupby('Ticker').size().min()}, "
      f"median={df_model.groupby('Ticker').size().median():.0f}, "
      f"max={df_model.groupby('Ticker').size().max()}")
print("Missing values:")
print(df_model.isnull().sum())
`;

const portableSetup = `# =============================================================================
# imports and config
# =============================================================================
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output during exploration
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
%matplotlib inline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Prefer the complete Kaggle dataset. The bundled project CSV is a local
# fallback for offline reruns or environments with a broken Kaggle client.
LOCAL_DATA_PATH = os.path.join('Dataset', 'cleaned_data.csv')
try:
    import kagglehub

    DATA_DIR = kagglehub.dataset_download(
        "eren2222/complete-indonesia-stock-exchange-idx-2000-2024"
    )
    DATA_PATH = os.path.join(DATA_DIR, "IDX 2000-2024.csv")
    print(f"Using Kaggle dataset CSV: {DATA_PATH}")
except Exception as exc:
    if not os.path.exists(LOCAL_DATA_PATH):
        raise
    DATA_PATH = LOCAL_DATA_PATH
    print(f"Kaggle dataset unavailable ({type(exc).__name__}); "
          f"using bundled project CSV: {DATA_PATH}")

# Project scope: only 2020-2024 per GOAL.md
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

print("=" * 70)
print("IDX NEXT-DAY PRICE DIRECTION PREDICTION")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version.split()[0]}")
print(f"Pandas: {pd.__version__}, NumPy: {np.__version__}")
print(f"Date range: {START_DATE} to {END_DATE}")
print("=" * 70)
`;

function splitSource(source) {
  return source.match(/[^\n]*\n|[^\n]+$/g) || [];
}

function replaceCell(notebook, marker, replacement, label) {
  const cell = notebook.cells.find(
    (candidate) =>
      candidate.cell_type === 'code' &&
      candidate.source.join('').includes(marker),
  );
  if (!cell) {
    throw new Error(`Could not find ${label}`);
  }
  cell.source = splitSource(replacement);
}

for (const path of notebooks) {
  const notebook = JSON.parse(fs.readFileSync(path, 'utf8'));
  const portableSetupNeedsRefresh = notebook.cells.some(
    (cell) =>
      cell.cell_type === 'code' &&
      cell.source.join('').includes('LOCAL_DATA_PATH = os.path.join('),
  );
  const legacySetup = notebook.cells.some(
    (cell) =>
      cell.cell_type === 'code' &&
      cell.source.join('').includes('DATA_DIR = kagglehub.dataset_download('),
  );
  if (portableSetupNeedsRefresh || legacySetup) {
    replaceCell(
      notebook,
      portableSetupNeedsRefresh
        ? 'LOCAL_DATA_PATH = os.path.join('
        : 'DATA_DIR = kagglehub.dataset_download(',
      portableSetup,
      `portable setup in ${path}`,
    );
  }
  replaceCell(
    notebook,
    "sector_df = pd.concat(sector_dfs, ignore_index=True)",
    sectorLoader,
    `sector loader in ${path}`,
  );
  const legacySummary = notebook.cells.some(
    (cell) =>
      cell.cell_type === 'code' &&
      cell.source
        .join('')
        .includes('print(f"Tickers:           {df_model["Ticker"].nunique()}")'),
  );
  if (legacySummary) {
    replaceCell(
      notebook,
      'print(f"Tickers:           {df_model["Ticker"].nunique()}")',
      exportSummary,
      `export summary in ${path}`,
    );
  }
  fs.writeFileSync(path, `${JSON.stringify(notebook, null, 1)}\n`);
  console.log(`Updated ${path}`);
}

for (const path of featureNotebooks) {
  const notebook = JSON.parse(fs.readFileSync(path, 'utf8'));
  const danglingDf = notebook.cells.find(
    (cell) => cell.cell_type === 'code' && cell.source.join('').trim() === 'df',
  );
  if (!danglingDf) {
    throw new Error(`Could not find dataframe display cell in ${path}`);
  }
  danglingDf.source = ['df\n'];

  if (path === 'Week2_submission_JCI_Prediction.ipynb') {
    for (const cell of notebook.cells) {
      if (
        cell.cell_type === 'code' &&
        cell.outputs.some((output) => output.output_type === 'error')
      ) {
        cell.outputs = [];
        cell.execution_count = null;
      }
    }
  }

  fs.writeFileSync(path, `${JSON.stringify(notebook, null, 1)}\n`);
  console.log(`Updated ${path}`);
}
