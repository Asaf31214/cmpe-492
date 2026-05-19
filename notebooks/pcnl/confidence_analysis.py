import polars as pl
from pathlib import Path
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"
files = sorted(RESULTS_DIR.glob("predictions_*.csv"))

CONF_MAP = {"low": 1, "medium": 2, "moderate": 2, "high": 3}

rows = []
for f in files:
    stem = f.stem.replace("predictions_", "")
    df = pl.read_csv(f).filter(pl.col("predicted").is_not_null())
    df = df.with_columns([
        pl.col("confidence").str.to_lowercase().replace(CONF_MAP).cast(pl.Int64, strict=False),
        (pl.col("predicted") == pl.col("actual")).alias("correct"),
    ]).filter(pl.col("confidence").is_not_null())
    for r in df.iter_rows(named=True):
        rows.append({
            "model": stem,
            "confidence": int(r["confidence"]),
            "correct": int(r["correct"]),
        })

rdf = pl.DataFrame(rows)

corr, p = spearmanr(rdf["confidence"], rdf["correct"])
print(f"Overall Spearman correlation (confidence vs correct): {corr:.3f} (p={p:.4e})")

print()
for model in sorted(rdf["model"].unique()):
    mdf = rdf.filter(pl.col("model") == model)
    corr, p = spearmanr(mdf["confidence"], mdf["correct"])
    print(f"{model}: Spearman={corr:.3f}, p={p:.4e}")
    for level in [1, 2, 3]:
        ldf = mdf.filter(pl.col("confidence") == level)
        if len(ldf) == 0:
            continue
        acc = ldf["correct"].mean()
        print(f"  {['low','medium','high'][level-1]} (n={len(ldf):>4}): accuracy={acc:.3f}")
    print()
