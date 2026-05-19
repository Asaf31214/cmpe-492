import json
import polars as pl
from ollama import chat
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
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
    "GUY SCORE",
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


class SyntheticPatient(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    YAŞ: int = Field(description="Age in years (0-18)")
    CİNSİYET: int = Field(description="1=Female, 2=Male")
    TARAF: int = Field(description="1=Right, 2=Left")
    ASA_SKORU: int = Field(alias="ASA SKORU", description="1=Healthy, 2=Mild, 3=Severe")
    BT: int = Field(description="0=No, 1=Yes")
    LOKALİZASYON: int = Field(description="1=Upper, 2=Middle, 3=Lower, 4=Pelvis, 5=Partial, 11=Staghorn, 20=Multi")
    ÖZGEÇMİŞ: int = Field(description="0=None, 1=DM, 2=Anticoag, 3=Pred, 4=Chronic, 5=CVS, 6=Other")
    GEÇİRİLMİŞ_CERRAHİ: int = Field(alias="GEÇİRİLMİŞ CERRAHİ", description="0=Primary, 1=PCNL, 2=ESWL, 3=Pyelo, 4=URS")
    TAS_ANAMNEZI: int = Field(alias="TAS ANAMNEZI", description="0=None, 1=Yes")
    İKAB: int = Field(description="0=Neg, 1=Pos")
    KREATİNİN: float = Field(description="Serum creatinine")
    SOLİTER_BB: int = Field(alias="SOLİTER BB", description="0=None, 1=Yes, 2=CRF")
    GUY_SCORE: int = Field(alias="GUY SCORE", description="1-4 (higher=complex)")
    TOPLAM_TAŞ_YÜKÜ_CM2: float = Field(alias="TOPLAM TAŞ YÜKÜ (CM2)", description="Stone load cm²")


def load_real_stats() -> dict:
    """Load statistics from real data to guide generation."""
    df = pl.read_csv(DATA_DIR / "processed.csv")

    stats = {
        "total": len(df),
        "success": len(df.filter(pl.col("SONUÇ-2") == 1)),
        "failure": len(df.filter(pl.col("SONUÇ-2") == 2)),
    }

    for col in PREOP_COLUMNS:
        if col in df.columns:
            vals = df[col].drop_nulls()
            if len(vals) > 0:
                stats[col] = {
                    "unique": vals.unique().to_list(),
                    "mean": vals.mean() if vals.dtype.is_numeric() else None,
                    "std": vals.std() if vals.dtype.is_numeric() else None,
                }

    failure_df = df.filter(pl.col("SONUÇ-2") == 2)
    stats["failure_profile"] = {}
    for col in PREOP_COLUMNS:
        if col in failure_df.columns:
            vals = failure_df[col].drop_nulls()
            if len(vals) > 0:
                stats["failure_profile"][col] = {
                    "mean": vals.mean() if vals.dtype.is_numeric() else None,
                    "mode": vals.mode().to_list()[0] if len(vals.mode()) > 0 else None,
                }

    return stats


def build_system_prompt(stats: dict, target_outcome: int) -> str:
    failure_profile = stats.get("failure_profile", {})

    profile_str = "\n".join(
        f"  - {col}: mean={v.get('mean')}, mode={v.get('mode')}"
        for col, v in failure_profile.items()
        if v.get("mean") is not None or v.get("mode") is not None
    )

    return f"""You are an expert urologist generating synthetic pediatric PCNL patient data for research.

## Task
Generate realistic synthetic patient records for pediatric patients (0-18 years) who underwent PCNL surgery.

## Target Outcome
Generate patients with outcome: {target_outcome} ({'Success (stone-free)' if target_outcome == 1 else 'Failure (residual fragments)'})

## Feature Reference
""" + "\n".join(f"- {c}: {COLUMN_DESC[c]}" for c in PREOP_COLUMNS) + f"""

## Real Data Statistics (for reference)
Total patients: {stats['total']}
- Success (stone-free): {stats['success']}
- Failure (residual): {stats['failure']}

## Failure Profile (patients with residual fragments)
{profile_str}

## Guidelines
1. Generate medically plausible combinations (e.g., higher GSS → larger stone burden)
2. Keep values within realistic ranges for pediatric patients
3. For failure cases: tend toward higher complexity (GSS 3-4, larger stones, anomalies)
4. For success cases: more varied, but generally lower complexity
5. Most patients should have no comorbidities (ÖZGEÇMİŞ=0, etc.)

Reply JSON array only, no other text."""


def generate_patients(target_outcome: int, n_samples: int, batch_size: int = 10) -> list[dict]:
    """Generate synthetic patients in batches."""
    stats = load_real_stats()
    system_prompt = build_system_prompt(stats, target_outcome)

    all_patients = []
    batches_needed = (n_samples + batch_size - 1) // batch_size

    print(f"Generating {n_samples} synthetic patients with outcome={target_outcome}")
    print(f"Batches needed: {batches_needed} (batch_size={batch_size})")

    for batch_idx in range(batches_needed):
        remaining = n_samples - len(all_patients)
        current_batch_size = min(batch_size, remaining)

        user_prompt = f"Generate exactly {current_batch_size} synthetic pediatric PCNL patients with outcome={target_outcome}. Return JSON array of patient objects."

        try:
            response = chat(
                model="qwen3.5:4b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                stream=False,
                think=False,
                options={"temperature": 0.7, "num_predict": 4096},
            )

            content = response["message"]["content"].strip()
            batch_data = json.loads(content)

            if isinstance(batch_data, list):
                for patient in batch_data:
                    if isinstance(patient, dict):
                        all_patients.append(patient)
            elif isinstance(batch_data, dict):
                all_patients.append(batch_data)

            print(f"Batch {batch_idx + 1}/{batches_needed}: Generated {len(batch_data) if isinstance(batch_data, list) else 1} patients")

        except Exception as e:
            print(f"Batch {batch_idx + 1} failed: {e}")

    return all_patients[:n_samples]


def validate_and_save(patients: list[dict], outcome: int, output_file: str):
    """Validate generated patients and save to CSV."""
    valid_patients = []

    for i, p in enumerate(patients):
        try:
            patient = SyntheticPatient(**p)
            valid_patients.append(
                {
                    **patient.model_dump(by_alias=True),
                    "SONUÇ-2": outcome,
                    "synthetic": 1,
                }
            )
        except Exception as e:
            print(f"Patient {i} invalid: {e}")
            continue

    if not valid_patients:
        print("No valid patients generated!")
        return

    df = pl.DataFrame(valid_patients)
    df.write_csv(DATA_DIR / output_file)
    print(f"\nSaved {len(valid_patients)} valid synthetic patients to {output_file}")

    print("\nGenerated data statistics:")
    for col in PREOP_COLUMNS:
        if col in df.columns:
            vals = df[col].drop_nulls()
            if len(vals) > 0:
                if vals.dtype.is_numeric():
                    print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}")
                else:
                    print(f"  {col}: mode={vals.mode().to_list()[0] if len(vals.mode()) > 0 else 'N/A'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic PCNL patient data")
    parser.add_argument("--outcome", type=int, default=2, choices=[1, 2], help="Target outcome (1=success, 2=failure)")
    parser.add_argument("--n", type=int, default=50, help="Number of synthetic patients to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Patients per batch")
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: synthetic_outcome_{n}.csv)")

    args = parser.parse_args()

    output_file = args.output or f"synthetic_outcome_{args.outcome}.csv"

    patients = generate_patients(
        target_outcome=args.outcome,
        n_samples=args.n,
        batch_size=args.batch_size,
    )

    validate_and_save(patients, args.outcome, output_file)


if __name__ == "__main__":
    main()
