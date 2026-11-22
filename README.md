# Obesity Risk & Lifestyle Advisor

โปรเจกต์ Machine Learning สำหรับประเมินระดับความเสี่ยงโรคอ้วนจากพฤติกรรมการกินและการออกกำลังกาย พร้อมแนะนำแนวทางปรับพฤติกรรม ผ่านเว็บแอป FastAPI
- 66050027 กันตวัฒน์ แช่มช้อย
- 66050053 จิรฑภูมิ ดอนคำไพร 
- 66050066 เจษฎา ภูครองทอง

## โครงสร้างโปรเจกต์

```text
.
├── ML_Project_Final.ipynb          # โน้ตบุ๊กหลักสำหรับเทรน ทดสอบ และเปรียบเทียบโมเดล
├── export_obesity_models.py        # สคริปต์รันโน้ตบุ๊กเพื่อ export โมเดลและผลวัดผล
├── obesity_web.py                  # FastAPI server และ API สำหรับทำนายผล
├── requirements.txt                # รายการแพ็กเกจ Python ที่ต้องใช้
├── .gitignore
├── models/
│   └── obesity_this_models.joblib  # ไฟล์โมเดลที่ export แล้ว (Model 1 + Model 2)
└── static/
    └── index.html                  # หน้าเว็บ UI
```

## การทำงานของโมเดล

ระบบแบ่งเป็น 2 ขั้นตอน โดยใช้ข้อมูลจาก Kaggle dataset [`aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster`](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster):

1. **Model 1 — Obesity Risk Classification (`Low` / `Medium` / `High`)**
   ใช้ 14 ฟีเจอร์ด้านข้อมูลพื้นฐานและพฤติกรรม (`Gender`, `Age`, `family_history_with_overweight`, `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`, `SCC`, `FAF`, `TUE`, `CALC`, `MTRANS`) โดยตัด `Weight`, `Height` และ `BMI` ออกเพื่อป้องกัน Data Leakage
2. **Model 2 — Lifestyle Recommendation**
   รับ 14 ฟีเจอร์เดิม ร่วมกับ `BMI` และผลทำนายจาก Model 1 (`ML1_Predicted_Risk`) เพื่อจัดกลุ่มคำแนะนำ 5 แบบ: `Nutrition Support`, `Improve Diet`, `Increase Activity`, `Improve Lifestyle`, `Maintain Habits`

### ผลวัดผลบนชุดข้อมูลทดสอบ

**Model 1 (เลือก LightGBM)**

| Model | CV F1 | Test F1 | Test Accuracy |
| --- | --- | --- | --- |
| Baseline (Majority) | 0.2118 | 0.2121 | 0.4665 |
| LogisticRegression | 0.6633 | 0.6681 | 0.6818 |
| RandomForest | 0.8266 | 0.8415 | 0.8517 |
| **LightGBM** | **0.8337** | **0.8568** | **0.8660** |

**Model 2 (เลือก LightGBM จาก CV Macro-F1)**

| Model | CV Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | --- | --- | --- |
| Baseline (Majority) | 0.1555 | 0.6045 | 0.1507 |
| LogisticRegression | 0.7703 | 0.8810 | 0.8054 |
| RandomForest | 0.8198 | 0.9582 | **0.9108** |
| **LightGBM** | **0.8745** | **0.9711** | 0.8881 |

## วิธีติดตั้งและรันเว็บแอป

ในโฟลเดอร์มีไฟล์ `models/obesity_this_models.joblib` (~3.4 MB) อยู่แล้ว สามารถติดตั้งแพ็กเกจและรันได้เลย:

```bash
pip install -r requirements.txt
uvicorn obesity_web:app --reload
```

เปิดเบราว์เซอร์ไปที่ `http://127.0.0.1:8000`

### กรณีต้องการเทรนและ Export โมเดลใหม่จากโน้ตบุ๊ก

```bash
python export_obesity_models.py
```

## API Endpoints

- `GET /` — หน้าเว็บ `static/index.html`
- `GET /api/metrics` — คืนค่าผลเปรียบเทียบโมเดล, Confusion Matrix และ Feature Importance ของ Model 1 & Model 2
- `POST /api/predict` — รับค่าฟีเจอร์ 14 ตัว พร้อม `Weight` (kg) และ `HeightCm` (cm) เพื่อคำนวณ BMI, ผลทำนายความเสี่ยงจาก Model 1 และคำแนะนำจาก Model 2
