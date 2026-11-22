"""Export Model 1 and Model 2 from ML_Project_Final.ipynb into models/obesity_this_models.joblib."""

import json
import os
import matplotlib
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")

nb = json.load(open("ML_Project_Final.ipynb", encoding="utf-8"))
ns = {"display": lambda *a, **k: None}

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "demo.launch(" in src or "demo_v2.launch(" in src:
        src = src.replace("demo.launch(share=True)", "pass").replace("demo_v2.launch(share=True)", "pass")
    src = src.replace("plt.show()", "plt.close('all')")
    if src.strip():
        exec(src, ns)

ml1_cms = {}
for m_name in ["LogisticRegression", "RandomForest", "LightGBM"]:
    ml1_cms[m_name] = confusion_matrix(ns["y_test"], ns["test_preds"][m_name]).tolist()

prep1 = ns["best_pipe"].named_steps["prep"]
mod1 = ns["best_pipe"].named_steps["model"]
feat_names = [f.replace("num__", "").replace("cat__", "") for f in prep1.get_feature_names_out()]
imps = mod1.feature_importances_ if hasattr(mod1, "feature_importances_") else np.mean(np.abs(mod1.coef_), axis=0)
top_idx = np.argsort(imps)[::-1][:10]
ml1_top_features = [{"feature": str(feat_names[i]), "importance": float(imps[i])} for i in top_idx]

ml2_labels = list(ns["REC_LABELS"])
ml2_cm = confusion_matrix(ns["y2_test"], ns["ml2_pred"], labels=ml2_labels).tolist()

os.makedirs("models", exist_ok=True)
artifact = {
    "ml1_pipe": ns["best_pipe"],
    "ml1_best_name": ns["best_name"],
    "ml1_results": ns["results_df"].to_dict(orient="records"),
    "ml1_cms": ml1_cms,
    "ml1_top_features": ml1_top_features,
    "class_names": ns["class_names"],
    "base_cols": ns["base_cols"],
    "ml2_pipe": ns["ml2_best_pipe"],
    "ml2_best_name": ns["ml2_best_name"],
    "ml2_results": ns["ml2_results"].to_dict(orient="records"),
    "ml2_cm": ml2_cm,
    "ml2_labels": ml2_labels,
    "ml2_features": ns["ML2_FEATURES"],
    "rec_thai": ns["REC_THAI"],
}

out_path = "models/obesity_this_models.joblib"
joblib.dump(artifact, out_path)
print("Exported:", out_path)
print("ML1 Best:", artifact["ml1_best_name"], "| ML2 Best:", artifact["ml2_best_name"])
