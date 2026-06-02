import json
import polars as pl
from ollama import chat
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

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
    "YAŞ": "Age in years (0-18)",
    "CİNSİYET": "1=Female, 2=Male",
    "TARAF": "1=Right, 2=Left",
    "ASA SKORU": "1=Healthy, 2=Mild systemic disease, 3=Severe systemic disease",
    "BT": "0=No, 1=Yes (preoperative CT)",
    "LOKALİZASYON": "1=Upper, 2=Middle, 3=Lower, 4=Pelvis, 5=Partial staghorn, 11=Staghorn, 20=Multi",
    "ÖZGEÇMİŞ": "0=None, 1=Diabetes, 2=Anticoagulant, 3=Prednisone, 4=Chronic disease, 5=CVS-HT, 6=Other",
    "GEÇİRİLMİŞ CERRAHİ": "0=Primary, 1=PCNL, 2=ESWL, 3=Pyelolithotomy, 4=URS",
    "TAS ANAMNEZI": "0=None, 1=Yes (history of stone passage/intervention)",
    "İKAB": "0=Negative, 1=Positive (urine culture)",
    "KREATİNİN": "Serum creatinine level",
    "SOLİTER BB": "0=None, 1=Yes, 2=Chronic renal failure",
    "RENAL ANOMALİ": "0=None, or anomaly name",
    "EK RENAL HASTALIK": "0=None, or disease name",
    "VUCUT ANOMALI": "0=None, or anomaly name (e.g., scoliosis)",
    "GUY SCORE": "1-4 (Guy's Stone Score, higher=more complex)",
    "SATAVA S.": "Satava intraoperative complication score",
    "TOPLAM TAŞ YÜKÜ (CM2)": "Total stone burden in cm²",
}

GSS_DEF = (
    "GSS1: Single/normal anatomy | "
    "GSS2: Multiple stones/anomaly | "
    "GSS3: Staghorn/normal | "
    "GSS4: Staghorn/anomaly"
)


class PredictionOutput(BaseModel):
    prediction: int = Field(description="1 for success (stone-free), 2 for failure (residual fragments)")
    reasoning: str = Field(description="A single short sentence explaining the prediction")


def build_system_prompt() -> str:
    cols = "\n".join(f"  - {c}: {COLUMN_DESC[c]}" for c in PREOP_COLUMNS)
    return f"""You are an expert urologist predicting PCNL (Percutaneous Nephrolithotomy) outcomes in pediatric patients.

Task: Predict whether the surgery will be successful (stone-free) or not.

Guy's Stone Score (GSS): {GSS_DEF}

Preoperative Features:
{cols}

Output: Return JSON with exactly two fields:
  - "prediction": 1 for success (stone-free), 2 for failure (residual fragments)
  - "reasoning": A single short sentence explaining your prediction

Reply JSON only, no other text."""


def predict_row(system_prompt: str, user_prompt: str) -> dict:
    response = chat(
        model="qwen3.5:4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=PredictionOutput.model_json_schema(),
        stream=False,
        think=False,
    )
    content = response["message"]["content"].strip()
    return json.loads(content)


def main():
    df = pl.read_csv(DATA_DIR / "processed.csv")

    system_prompt = build_system_prompt()

    results = []

    print("Running LLM predictions with qwen3.5:4b...")
    print(f"Total samples: {len(df)}\n")

    for idx, row in tqdm(enumerate(df.iter_rows(named=True)), total=len(df), desc="Predicting"):
        features = ", ".join(f"{c}: {row.get(c, 'N/A')}" for c in PREOP_COLUMNS)

        try:
            prediction = predict_row(system_prompt, features)
            pred = prediction.get("prediction")
            reasoning = prediction.get("reasoning", "")
        except Exception as e:
            pred = None
            reasoning = f"Error: {e}"

        actual = row["SONUÇ-2"]
        results.append(
            {
                "row_idx": idx,
                "actual": actual,
                "predicted": pred,
                "reasoning": reasoning,
                "correct": 1 if pred == actual else 0,
            }
        )

    results_df = pl.DataFrame(results)
    results_df.write_csv(DATA_DIR / "llm_predictions_qwen.csv")

    valid_results = results_df.filter(pl.col("predicted").is_not_null())

    if len(valid_results) == 0:
        print("\n❌ No valid predictions made.")
        return

    y_true = valid_results["actual"].to_list()
    y_pred = valid_results["predicted"].to_list()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Total samples:        {len(results_df)}")
    print(f"Valid predictions:    {len(valid_results)}")
    print(f"Failed predictions:   {len(results_df) - len(valid_results)}")
    print("-" * 60)
    print(f"Accuracy:             {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1 Score (weighted):  {f1:.4f}")
    print("-" * 60)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["1=Success", "2=Failure"], digits=4, zero_division=0
        )
    )
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\nPredictions saved to: data/tabular/llm_predictions_qwen.csv")


if __name__ == "__main__":
    main()
