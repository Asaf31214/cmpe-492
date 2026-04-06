# Feature Dictionary: Pediatric PCNL Preoperative Features

## Target Variable

### sonuc_2 (Outcome)
Binary surgical outcome.
- `1` = Stone-free (success)
- `2` = Residual fragments (failure)

---

## Demographics

### age
Patient age in years.

### gender
Patient biological sex.
- `1` = Female
- `2` = Male

---

## Stone Characteristics

### side
Operated kidney side.
- `1` = Right
- `2` = Left

### stone_localization
Stone location within the kidney collecting system.
- `1` = Upper calyx
- `2` = Middle calyx
- `3` = Lower calyx
- `4` = Renal pelvis
- `5` = Partial staghorn
- `11` = Complete staghorn
- `20` = Multi-localization
- May also contain text: `"UPJ"` (ureteropelvic junction), `"üreter"` (ureter), or suffixed forms like `"1 (USG)"`.

### stone_burden_cm2
Total stone surface area in cm².

### ct_scan
Whether preoperative CT imaging was performed.
- `0` = No
- `1` = Yes

---

## Patient History

### comorbidity
Patient's pre-existing medical conditions.
- `0` = None
- `1` = Diabetes mellitus
- `2` = Anticoagulant use
- `3` = Prednisone use
- `4` = Chronic disease
- `5` = Cardiovascular disease / Hypertension
- `6` = Other

### previous_surgery
History of previous urological stone interventions. Decimal values encode combined procedures (e.g. `1.2` = PCNL + ESWL, `2.4` = ESWL + URS).
- `0` = Primary (no previous surgery)
- `1` = PCNL
- `2` = ESWL
- `3` = Pyelolithotomy
- `4` = URS
- `6` = Other

### stone_anamnesis
History of spontaneous stone passage or previous stone-related interventions.
- `0` = None
- `1` = Yes

### asa_score
ASA (American Society of Anesthesiologists) physical status classification.
- `1` = Healthy patient
- `2` = Mild systemic disease
- `3` = Severe systemic disease

---

## Laboratory & Anatomy

### urine_culture
Preoperative urine culture result. When positive, may contain the actual bacteria name. May also contain data entry errors (`iç sunucu hatası`).
- `0` = Negative
- `1` = Positive (unspecified)

### creatinine
Preoperative serum creatinine level (mg/dL). May contain non-numeric entries (`N`, `yok`).

### solitary_kidney
Solitary kidney or chronic renal failure status.
- `0` = None
- `1` = Solitary kidney
- `2` = Chronic renal failure

### renal_anomaly
Congenital renal anatomical anomaly.
- `0` = None
- `1` = Yes (unspecified)
- May also contain text: `"ATNALI"` (horseshoe kidney), `"UPJ DARLIĞI"` (UPJ obstruction), `"sağ atrofik"` (atrophic kidney).

### additional_renal_disease
Additional renal pathology.
- `0` = None
- `1` = Yes

