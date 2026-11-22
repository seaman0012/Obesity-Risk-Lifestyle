"""FastAPI web app serving Model 1 (Obesity Risk) + Model 2 (Lifestyle Recommendation)."""

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_PATH = "models/obesity_this_models.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run export_obesity_models.py first.")

artifact = joblib.load(MODEL_PATH)
ml1_pipe = artifact["ml1_pipe"]
ml1_best_name = artifact["ml1_best_name"]
ml1_results = artifact["ml1_results"]
class_names = artifact["class_names"]
base_cols = artifact["base_cols"]

ml2_pipe = artifact["ml2_pipe"]
ml2_best_name = artifact["ml2_best_name"]
ml2_results = artifact["ml2_results"]
ml2_features = artifact["ml2_features"]
rec_thai = artifact["rec_thai"]

REC_DETAILS = {
    "Nutrition Support": "ทานอาหารให้ครบมื้อและเพียงพอต่อพลังงานที่ใช้ หากน้ำหนักต่ำต่อเนื่องควรปรึกษานักกำหนดอาหาร",
    "Improve Diet": "ปรับมื้ออาหารให้สมดุล เพิ่มผักและอาหารที่หลากหลายโดยไม่ต้องอดอาหาร",
    "Increase Activity": "ค่อย ๆ เพิ่มการเคลื่อนไหวหรือออกกำลังกายให้สม่ำเสมอ และลดการนั่งหน้าจอนาน ๆ",
    "Improve Lifestyle": "ดื่มน้ำให้เพียงพอ พักผ่อนให้เป็นเวลา และลดแอลกอฮอล์หรือบุหรี่",
    "Maintain Habits": "พฤติกรรมโดยรวมอยู่ในเกณฑ์ดีแล้ว ดูแลต่อเนื่องได้ตามปกติ",
}

app = FastAPI(title="Obesity Risk & Lifestyle Recommendation")


class PredictRequest(BaseModel):
    Gender: str = "Male"
    Age: float = Field(24, ge=20, le=80)
    Weight: float = Field(70, gt=0, le=300)
    HeightCm: float = Field(170, gt=50, le=250)
    family_history_with_overweight: str = "yes"
    FAVC: str = "yes"
    FCVC: float = 2.0
    NCP: float = 3.0
    CAEC: str = "Sometimes"
    SMOKE: str = "no"
    CH2O: float = 2.0
    SCC: str = "no"
    FAF: float = 1.0
    TUE: float = 1.0
    CALC: str = "Sometimes"
    MTRANS: str = "Public_Transportation"


def classify_bmi_group(bmi: float) -> tuple[str, str]:
    if bmi < 18.5:
        return "น้ำหนักต่ำกว่าเกณฑ์", "underweight"
    if bmi < 25.0:
        return "เกณฑ์มาตรฐาน", "normal"
    if bmi < 30.0:
        return "น้ำหนักเกิน", "overweight"
    return "เกณฑ์โรคอ้วน", "obese"


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/metrics")
def get_metrics():
    return {
        "ml1_best": ml1_best_name,
        "ml1_results": ml1_results,
        "ml1_cms": artifact.get("ml1_cms", {}),
        "ml1_class_names": class_names,
        "ml1_top_features": artifact.get("ml1_top_features", []),
        "ml2_best": ml2_best_name,
        "ml2_results": ml2_results,
        "ml2_cm": artifact.get("ml2_cm", []),
        "ml2_labels": artifact.get("ml2_labels", []),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    bmi = req.Weight / ((req.HeightCm / 100.0) ** 2)
    if not np.isfinite(bmi) or bmi <= 0:
        raise HTTPException(status_code=400, detail="Invalid Weight or Height")

    row = req.model_dump()
    row["BMI"] = bmi
    raw_df = pd.DataFrame([row])

    ml1_probs = ml1_pipe.predict_proba(raw_df[base_cols])[0]
    ml1_idx = int(np.argmax(ml1_probs))
    ml1_risk = class_names[ml1_idx]

    row["ML1_Predicted_Risk"] = ml1_risk
    ml2_df = pd.DataFrame([row])[ml2_features]
    ml2_probs = ml2_pipe.predict_proba(ml2_df)[0]
    ml2_pred = str(ml2_pipe.predict(ml2_df)[0])

    bmi_label, bmi_tier = classify_bmi_group(bmi)
    return {
        "bmi": round(bmi, 1),
        "bmi_label": bmi_label,
        "bmi_tier": bmi_tier,
        "ml1": {
            "model": ml1_best_name,
            "prediction": ml1_risk,
            "confidence": round(float(ml1_probs[ml1_idx]) * 100, 1),
            "probs": {class_names[i]: round(float(ml1_probs[i]) * 100, 1) for i in range(len(class_names))},
        },
        "ml2": {
            "model": ml2_best_name,
            "prediction": ml2_pred,
            "title_th": rec_thai.get(ml2_pred, ml2_pred),
            "detail_th": REC_DETAILS.get(ml2_pred, ""),
            "probs": {str(lbl): round(float(p) * 100, 1) for lbl, p in zip(ml2_pipe.classes_, ml2_probs)},
        },
    }
