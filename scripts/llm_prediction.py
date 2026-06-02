import json
import polars as pl
from ollama import chat
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "tabular"

PREOP_COLUMNS = [
    "YAŞ",
    "CİNSİYET",
    "TARAF",
    "ASA SKORU",
    "BT",
    "LOKALİZASYON",
    "ÖZGEÇMİŞ",
    "GEÇİRİLMİŞ CERRAHİ",
    "TAS ANAMNEZI",
    "İKAB",
    "KREATİNİN",
    "SOLİTER BB",
    "RENAL ANOMALİ",
    "EK RENAL HASTALIK",
    "VUCUT ANOMALI",
    "GUY SCORE",
    "SATAVA S.",
    "TOPLAM TAŞ YÜKÜ (CM2)",
]

COLUMN_DESC = {
    "YAŞ": "Age",
    "CİNSİYET": "1=Female,2=Male",
    "TARAF": "1=Right,2=Left",
    "ASA SKORU": "1=Healthy,2=Mild,3=Severe",
    "BT": "0=No,1=Yes",
    "LOKALİZASYON": "1=Upper,2=Middle,3=Lower,4=Pelvis,5=Partial,11=Staghorn,20=Multi",
    "ÖZGEÇMİŞ": "0=None,1=DM,2=Anticoag,3=Pred,4=Chronic,5=CVS,6=Other",
    "GEÇİRİLMİŞ CERRAHİ": "0=Primary,1=PCNL,2=ESWL,3=Pyelo,4=URS",
    "TAS ANAMNEZI": "0=None,1=Yes",
    "İKAB": "0=Neg,1=Pos",
    "KREATİNİN": "Creatinine",
    "SOLİTER BB": "0=None,1=Yes,2=CRF",
    "RENAL ANOMALİ": "0=None",
    "EK RENAL HASTALIK": "0=None",
    "VUCUT ANOMALI": "0=None",
    "GUY SCORE": "1-4 (higher=complex)",
    "SATAVA S.": "Satava",
    "TOPLAM TAŞ YÜKÜ (CM2)": "Stone load cm²",
}

GSS_DEF = "GSS1:Single/normal GSS2:Multi/anomaly GSS3:Staghorn/normal GSS4:Staghorn/anomaly"

def load_column_summary() -> dict:
    df = pl.read_csv(DATA_DIR / "column_summary.csv")
    result = {}
    for row in df.iter_rows(named=True):
        if row["column"] in PREOP_COLUMNS:
            result[row["column"]] = {"unique": row["unique_count"], "null_pct": row["null_pct"]}
    return result

def build_system_prompt(col_summary: dict) -> str:
    cols = "\n".join(f"- {c}: {COLUMN_DESC[c]}" for c in PREOP_COLUMNS)
    return f"""Predict PCNL outcome (1=success,2=fail).

GSS: {GSS_DEF}

Features:
{cols}

Reply JSON only: {{"prediction":1|2,"confidence":0-1}}"""

def predict_row(client, system_prompt: str, user_prompt: str) -> dict:
    response = client(
        model="dolphin3:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        stream=False,
        options={"num_predict": 80, "temperature": 0.3},
    )
    content = response["message"]["content"].strip()
    if not content:
        return {"prediction": None, "confidence": 0}
    return json.loads(content)

def main():
    df = pl.read_csv(DATA_DIR / "processed.csv")
    col_summary = load_column_summary()
    system_prompt = build_system_prompt(col_summary)

    results = []
    for idx, row in enumerate(df.iter_rows(named=True)):
        features = ", ".join(f"{c}: {row.get(c, 'N/A')}" for c in PREOP_COLUMNS)
        try:
            prediction = predict_row(chat, system_prompt, features)
            pred = prediction.get("prediction")
        except Exception as e:
            print(f"Error row {idx}: {e}")
            pred = None

        actual = row["SONUÇ-2"]
        results.append({
            "row_idx": idx,
            "actual": actual,
            "predicted": pred,
            "confidence": prediction.get("confidence") if pred else 0,
            "correct": 1 if pred == actual else 0,
        })
        print(f"{idx + 1}/{len(df)}: Actual={actual}, Pred={pred}, OK={pred == actual}")

    results_df = pl.DataFrame(results)
    results_df.write_csv(DATA_DIR / "llm_predictions.csv")
    correct = results_df.filter(pl.col("predicted").is_not_null())["correct"].sum()
    total = results_df.filter(pl.col("predicted").is_not_null()).height
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}" if total else "\nNo predictions")

if __name__ == "__main__":
    main()
