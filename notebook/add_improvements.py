import json
import copy

notebook_path = r"d:\Users\BuiTrongNguyen\Student Dropout and Academic Success\notebook\hello.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ─── Helper: build a code cell ───────────────────────────────────────────────

def code_cell(cell_id, source_lines):
    """source_lines: list of Python source lines (each should end with \\n except last)"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def md_cell(cell_id, text):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [text],
    }

# ─── 1. Find insertion point: after "## Feature Construction" markdown ────────

feat_construction_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and "Feature Construction" in src:
        feat_construction_idx = i
        break

if feat_construction_idx is None:
    print("❌ Could not find 'Feature Construction' markdown cell!")
else:
    feat_eng_cell = code_cell("feat_eng_new_01", [
        "# ===== FEATURE ENGINEERING: Tạo thêm Features Mới =====\n",
        "# Các features mới được tạo từ sự kết hợp của các features hiện có\n",
        "\n",
        "# 1. Stress Score: tổng hợp các yếu tố stress chính\n",
        "df['stress_score'] = df['academic_pressure'] + df['work_pressure'] + df['financial_stress']\n",
        "\n",
        "# 2. Satisfaction Score: tổng hợp mức độ hài lòng\n",
        "df['satisfaction_score'] = df['study_satisfaction'] + df['job_satisfaction']\n",
        "\n",
        "# 3. Khoảng cách giữa áp lực học và sự hài lòng (cao = nguy hiểm)\n",
        "df['pressure_satisfaction_gap'] = df['academic_pressure'] - df['study_satisfaction']\n",
        "\n",
        "# 4. Số giờ làm/học so với thời gian ngủ\n",
        "df['work_hours_per_sleep'] = df['work/study_hours'] / (df['sleep_duration'] + 1)\n",
        "\n",
        "# 5. Tỷ lệ CGPA so với áp lực học tập\n",
        "df['cgpa_pressure_ratio'] = df['cgpa'] / (df['academic_pressure'] + 1)\n",
        "\n",
        "print('Shape sau Feature Engineering:', df.shape)\n",
        "df.head()",
    ])
    cells.insert(feat_construction_idx + 1, feat_eng_cell)
    print(f"✅ Inserted Feature Engineering cell after index {feat_construction_idx}")

# ─── 2. Change X_final = X[top_features] → X_final = X ───────────────────────

changed_xfinal = False
for cell in cells:
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", [])
    new_src = []
    for line in src:
        if "X_final = X[top_features]" in line:
            new_src.append("X_final = X  # Sử dụng toàn bộ features (bao gồm cả features mới)\n")
            changed_xfinal = True
        else:
            new_src.append(line)
    cell["source"] = new_src

if changed_xfinal:
    print("✅ Changed X_final to use all features")
else:
    print("❌ Could not find X_final = X[top_features] line")

# ─── 3. Find insertion point: after XGBoost model cell (id: 40920eea) ─────────

xgb_cell_idx = None
for i, cell in enumerate(cells):
    if cell.get("id") == "40920eea":
        xgb_cell_idx = i
        break

if xgb_cell_idx is None:
    # Fallback: find by content
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "XGBClassifier" in src and "xgb_model.fit" in src and "GridSearchCV" not in src:
                xgb_cell_idx = i
                break

if xgb_cell_idx is None:
    print("❌ Could not find XGBoost model cell!")
else:
    lgb_cell = code_cell("lgbm_model_new_01", [
        "from lightgbm import LGBMClassifier\n",
        "\n",
        "# LightGBM - thường cho accuracy cao hơn XGBoost và RF trên tabular data\n",
        "lgb_model = LGBMClassifier(\n",
        "    n_estimators=500,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.05,\n",
        "    num_leaves=31,\n",
        "    subsample=0.8,\n",
        "    colsample_bytree=0.8,\n",
        "    random_state=42,\n",
        "    verbose=-1\n",
        ")\n",
        "lgb_model.fit(X_train, y_train)\n",
        "y_train_pred_lgb = lgb_model.predict(X_train)\n",
        "y_test_pred_lgb = lgb_model.predict(X_test)\n",
        "\n",
        "evaluate_model(y_train, y_train_pred_lgb, 'Train (LightGBM)')\n",
        "evaluate_model(y_test, y_test_pred_lgb, 'Test (LightGBM)')",
    ])
    cells.insert(xgb_cell_idx + 1, lgb_cell)
    print(f"✅ Inserted LightGBM cell after index {xgb_cell_idx}")

# ─── 4. Find last VotingClassifier cell and insert StackingClassifier after ───

last_voting_idx = None
for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "VotingClassifier" in src and "evaluate_model" in src:
            last_voting_idx = i

if last_voting_idx is None:
    print("❌ Could not find VotingClassifier cell!")
else:
    stacking_cell = code_cell("stacking_clf_new_01", [
        "from sklearn.ensemble import StackingClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "# Stacking Ensemble: mạnh hơn Voting Classifier\n",
        "base_estimators = [\n",
        "    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,\n",
        "                                   min_samples_split=10, random_state=42)),\n",
        "    ('xgb', XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)),\n",
        "    ('lgb', LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,\n",
        "                           num_leaves=31, random_state=42, verbose=-1)),\n",
        "]\n",
        "\n",
        "stacking = StackingClassifier(\n",
        "    estimators=base_estimators,\n",
        "    final_estimator=LogisticRegression(max_iter=1000),\n",
        "    cv=5,\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "stacking.fit(X_train, y_train)\n",
        "y_train_pred_stack = stacking.predict(X_train)\n",
        "y_test_pred_stack = stacking.predict(X_test)\n",
        "\n",
        "evaluate_model(y_train, y_train_pred_stack, 'Train (Stacking Ensemble)')\n",
        "evaluate_model(y_test, y_test_pred_stack, 'Test (Stacking Ensemble)')",
    ])
    cells.insert(last_voting_idx + 1, stacking_cell)
    print(f"✅ Inserted StackingClassifier cell after index {last_voting_idx}")

# ─── Save ─────────────────────────────────────────────────────────────────────

nb["cells"] = cells
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✅ Notebook saved successfully!")
print(f"Total cells: {len(cells)}")
