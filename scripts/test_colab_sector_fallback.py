import json
from pathlib import Path

import pandas as pd


NOTEBOOK = Path("Week2_submission_JCI_Prediction.ipynb")
SOURCE_FOLDER = "os.path.join('Dataset', 'stock-list-per-sector')"
MISSING_FOLDER = "os.path.join('Dataset', '__missing_sector_folder__')"


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
source = "".join(notebook["cells"][27]["source"])
source = source.replace(SOURCE_FOLDER, MISSING_FOLDER)

namespace = {
    "os": __import__("os"),
    "pd": pd,
    "unique_tickers": {
        ticker
        for ticker in source.replace("\n", " ").split("'")
        if ticker.isupper() and ticker.isalpha() and 4 <= len(ticker) <= 5
    },
}
exec(source, namespace)

sector_df = namespace["sector_df"]
counts = sector_df["Sector"].value_counts().to_dict()
assert counts == {"Energy": 91, "Technology": 47}, counts
assert sector_df["Ticker"].nunique() == 138
print("Colab sector fallback passed:", counts)
