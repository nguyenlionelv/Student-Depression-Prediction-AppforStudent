"""
Rule-based logic translator: maps SHAP top features to Vietnamese explanations
and provides personalized recommendations.
"""
from typing import List, Tuple

# ─── Feature label mapping (English key → Vietnamese label) ───────────────────
FEATURE_LABELS = {
    "academic_pressure":        "Áp lực học tập",
    "work_pressure":            "Áp lực công việc",
    "cgpa":                     "Điểm GPA",
    "study_satisfaction":       "Mức độ hài lòng học tập",
    "job_satisfaction":         "Mức độ hài lòng công việc",
    "sleep_duration":           "Thời gian ngủ",
    "work_study_hours":         "Số giờ học/làm",
    "financial_stress":         "Áp lực tài chính",
    "family_history":           "Tiền sử gia đình",
    "suicidal_thoughts":        "Suy nghĩ tiêu cực",
    "gender":                   "Giới tính",
    "age":                      "Độ tuổi",
    "degree":                   "Bậc học",
    "stress_score":             "Chỉ số stress tổng hợp",
    "satisfaction_score":       "Chỉ số hài lòng tổng hợp",
    "pressure_satisfaction_gap":"Khoảng cách áp lực - hài lòng",
    "work_hours_per_sleep":     "Giờ học/làm trên giờ ngủ",
    "cgpa_pressure_ratio":      "Tỷ lệ GPA/áp lực",
    "dietary_habits_Healthy":   "Thói quen ăn uống lành mạnh",
    "dietary_habits_Moderate":  "Thói quen ăn uống bình thường",
    "dietary_habits_Others":    "Thói quen ăn uống khác",
    "dietary_habits_Unhealthy": "Thói quen ăn uống không lành mạnh",
}

# ─── Feature-level comments (positive SHAP = increases depression risk) ────────
FEATURE_COMMENTS = {
    "academic_pressure": {
        "increases": "Áp lực học tập cao đang tác động tiêu cực đáng kể đến sức khỏe tâm thần của bạn.",
        "decreases": "Mức áp lực học tập vừa phải là yếu tố tích cực cho sức khỏe tâm thần.",
    },
    "work_pressure": {
        "increases": "Áp lực công việc cao đang góp phần gia tăng nguy cơ trầm cảm.",
        "decreases": "Áp lực công việc ở mức kiểm soát được giúp giảm nguy cơ trầm cảm.",
    },
    "cgpa": {
        "increases": "Kết quả học tập chưa tốt có thể gây lo lắng và mất tự tin kéo dài.",
        "decreases": "Thành tích học tập tốt đang đóng vai trò như một yếu tố bảo vệ tâm lý tích cực.",
    },
    "study_satisfaction": {
        "increases": "Mức độ không hài lòng với việc học tập đang ảnh hưởng tiêu cực đến tâm trạng.",
        "decreases": "Sự hài lòng với việc học là yếu tố quan trọng bảo vệ sức khỏe tâm thần.",
    },
    "job_satisfaction": {
        "increases": "Sự không hài lòng với công việc/thực tập đang gây ra căng thẳng tích lũy.",
        "decreases": "Sự hài lòng trong công việc giúp cân bằng tâm lý hiệu quả.",
    },
    "sleep_duration": {
        "increases": "Thời gian ngủ không đủ đang ảnh hưởng xấu đến sức khỏe tâm thần.",
        "decreases": "Thói quen ngủ tốt đang hỗ trợ tích cực cho sức khỏe tâm thần.",
    },
    "financial_stress": {
        "increases": "Áp lực tài chính cao là một yếu tố gây căng thẳng tâm lý nghiêm trọng.",
        "decreases": "Tình hình tài chính ổn định giúp giảm bớt căng thẳng tâm lý.",
    },
    "suicidal_thoughts": {
        "increases": "Đây là dấu hiệu cần được chú ý và tư vấn từ chuyên gia sức khỏe tâm thần ngay.",
        "decreases": "Không có suy nghĩ tiêu cực là dấu hiệu tích cực.",
    },
    "work_study_hours": {
        "increases": "Số giờ học/làm quá nhiều dẫn đến kiệt sức (burnout) và ảnh hưởng tiêu cực.",
        "decreases": "Thời lượng học/làm hợp lý giúp duy trì sự cân bằng sức khỏe.",
    },
    "family_history": {
        "increases": "Tiền sử gia đình về sức khỏe tâm thần làm tăng nguy cơ trầm cảm.",
        "decreases": "Không có tiền sử gia đình là yếu tố bảo vệ quan trọng.",
    },
    "stress_score": {
        "increases": "Tổng mức stress từ nhiều nguồn (học, việc, tài chính) đang quá cao.",
        "decreases": "Mức stress tổng hợp đang ở mức kiểm soát được.",
    },
    "satisfaction_score": {
        "increases": "Mức độ hài lòng tổng thể thấp đang tạo ra vòng lặp tiêu cực về tâm lý.",
        "decreases": "Mức độ hài lòng tổng thể cao giúp duy trì tâm lý tích cực.",
    },
    "pressure_satisfaction_gap": {
        "increases": "Khoảng cách lớn giữa áp lực và sự hài lòng tạo ra mất cân bằng tâm lý.",
        "decreases": "Sự cân bằng giữa áp lực và hài lòng đang ổn định.",
    },
    "work_hours_per_sleep": {
        "increases": "Tỷ lệ giờ làm việc so với giờ ngủ quá cao dẫn đến kiệt sức mãn tính.",
        "decreases": "Tỷ lệ làm việc/ngủ cân đối giúp phục hồi sức khỏe tốt.",
    },
    "dietary_habits_Unhealthy": {
        "increases": "Thói quen ăn uống không lành mạnh là yếu tố nguy cơ tiêu cực.",
        "decreases": "Không có ảnh hưởng từ thói quen ăn uống không lành mạnh.",
    },
    "dietary_habits_Healthy": {
        "increases": "Chế độ ăn lành mạnh đang hỗ trợ sức khỏe tâm thần.",
        "decreases": "Chế độ ăn lành mạnh là yếu tố bảo vệ tích cực.",
    },
    "gender": {
        "increases": "Các yếu tố liên quan đến áp lực giới tính hoặc kỳ vọng xã hội đang có ảnh hưởng tiêu cực.",
        "decreases": "Đặc điểm giới tính của bạn phản ánh khả năng thích ứng tâm lý tích cực trong mô hình.",
    },
    "age": {
        "increases": "Độ tuổi hiện tại của bạn đang nằm trong nhóm dễ đối mặt với nhiều áp lực và thay đổi tâm lý.",
        "decreases": "Độ tuổi hiện tại của bạn nằm trong nhóm cho thấy sự ổn định về mặt tâm lý.",
    },
    "degree": {
        "increases": "Chương trình học hiện tại đòi hỏi cường độ cao, có thể là một phần nguyên nhân gây căng thẳng.",
        "decreases": "Chương trình học hiện tại không tạo ra áp lực quá lớn lên tâm lý của bạn.",
    },
    "cgpa_pressure_ratio": {
        "increases": "Sự mất cân bằng giữa kết quả học tập và áp lực quá lớn đang gây ra trạng thái căng thẳng.",
        "decreases": "Sự cân đối giữa điểm số và nỗ lực học tập đang duy trì trạng thái tâm lý tích cực.",
    },
    "dietary_habits_Moderate": {
        "increases": "Chế độ ăn uống hiện tại chưa thực sự tối ưu, có thể tác động đến sự phục hồi tâm lý.",
        "decreases": "Thói quen ăn uống của bạn đang duy trì ở mức ổn định.",
    },
    "dietary_habits_Others": {
        "increases": "Chế độ ăn uống hiện tại chưa thực sự tối ưu, có thể tác động đến sự phục hồi tâm lý.",
        "decreases": "Thói quen ăn uống của bạn đang duy trì ở mức khá ổn định.",
    },
}

# ─── Recommendation bank ──────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "high_risk": [
        " Hãy tìm kiếm sự hỗ trợ từ chuyên gia tư vấn tâm lý hoặc bác sĩ ngay.",
        " Gọi đường dây hỗ trợ sức khỏe tâm thần: 1800 599 920 (miễn phí, 24/7).",
        " Chia sẻ cảm xúc với người thân, bạn bè đáng tin cậy.",
    ],
    "medium_risk": [
        " Thực hành thiền hoặc các bài tập thư giãn 10-15 phút mỗi ngày.",
        " Lên kế hoạch học tập hợp lý để tránh áp lực dồn dập.",
        " Cân nhắc tham gia nhóm tư vấn tâm lý của trường/cơ quan.",
    ],
    "low_risk": [
        " Duy trì thói quen sinh hoạt lành mạnh như hiện tại.",
        " Tiếp tục cân bằng giữa học tập, công việc và nghỉ ngơi.",
    ],
    "sleep": [
        " Cố gắng ngủ đủ 7-8 tiếng mỗi ngày. Tắt điện thoại trước khi ngủ 30 phút.",
    ],
    "financial": [
        " Tìm hiểu về học bổng, hỗ trợ tài chính của trường hoặc tổ chức xã hội.",
    ],
    "academic": [
        " Chia nhỏ mục tiêu học tập. Hỏi giảng viên / bạn bè khi cần hỗ trợ.",
    ],
    "social": [
        " Dành thời gian giao lưu xã hội, tham gia câu lạc bộ hoặc hoạt động tình nguyện.",
    ],
    "exercise": [
        " Tập thể dục ít nhất 30 phút mỗi ngày — có thể giảm đến 40% nguy cơ trầm cảm.",
    ],
    "diet": [
        " Cải thiện chế độ ăn. Hạn chế thức ăn nhanh, tăng rau củ và uống đủ nước.",
    ],
}


def get_risk_level(probability: float) -> str:
    if probability < 0.2:
        return "Rất thấp"
    elif probability < 0.4:
        return "Ít khả năng"
    elif probability < 0.6:
        return "Có nguy cơ"
    elif probability < 0.8:
        return "Tương đối cao"
    else:
        return "Rất cao"


def get_impact_level(abs_shap: float, max_shap: float) -> str:
    ratio = abs_shap / (max_shap + 1e-9)
    if ratio > 0.6:
        return "cao"
    elif ratio > 0.3:
        return "trung bình"
    return "thấp"


def translate_factors(shap_values, feature_names: List[str], top_k: int = 5):
    """
    Returns top-k factors sorted by absolute SHAP value.
    shap_values: 1D array of shap values for one sample.
    """
    from src.schemas import FactorDetail
    import numpy as np

    pairs = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_pairs = pairs[:top_k]
    max_abs = max(abs(v) for _, v in top_pairs) if top_pairs else 1.0

    factors = []
    for feat, sv in top_pairs:
        direction = "tăng nguy cơ" if sv > 0 else "giảm nguy cơ"
        key = "increases" if sv > 0 else "decreases"
        comment_map = FEATURE_COMMENTS.get(feat, {})
        # (we return comment separately in explanation, not in FactorDetail)
        factors.append(
            FactorDetail(
                feature=feat,
                label_vi=FEATURE_LABELS.get(feat, feat),
                direction=direction,
                impact_level=get_impact_level(abs(sv), max_abs),
                shap_value=round(float(sv), 4),
            )
        )
    return factors


def build_explanation(
    factors,
    risk_level: str,
    probability: float,
    input_data: dict,
) -> Tuple[str, List[str]]:
    """Build a Vietnamese explanation paragraph and recommendation list."""

    # Opening sentence
    prob_pct = int(probability * 100)
    opening = (
        f"Dựa trên thông tin bạn cung cấp, mô hình đánh giá xác suất trầm cảm của bạn là "
        f"khoảng {prob_pct}%, mức độ nguy cơ: <strong>{risk_level}</strong>.<br><br>"
    )

    # Factor sentences (top 3 that increase risk)
    increase_factors = [f for f in factors if "tăng" in f.direction][:3]
    factor_text = ""
    bullets = []
    for f in increase_factors:
        feat_key = f.feature
        comment = FEATURE_COMMENTS.get(feat_key, {}).get("increases", "")
        if not comment:
            comment = f"Yếu tố {f.label_vi.lower()} (mức {f.impact_level}) đang góp phần gia tăng nguy cơ."
        bullets.append(f"• {comment}<br>")
        
    if bullets:
        factor_text = "Các yếu tố chính ảnh hưởng tiêu cực:<br>" + "\n".join(bullets)

    # Recommendations
    recs = []
    if risk_level in ("Tương đối cao", "Rất cao"):
        recs += RECOMMENDATIONS["high_risk"]
    elif risk_level in ("Có nguy cơ", "Ít khả năng"):
        recs += RECOMMENDATIONS["medium_risk"]
    else:
        recs += RECOMMENDATIONS["low_risk"]

    # Contextual recommendations
    if input_data.get("sleep_duration", 2) <= 1:
        recs += RECOMMENDATIONS["sleep"]
    if input_data.get("financial_stress", 0) >= 4:
        recs += RECOMMENDATIONS["financial"]
    if input_data.get("academic_pressure", 0) >= 4:
        recs += RECOMMENDATIONS["academic"]
    if input_data.get("work_study_hours", 0) >= 10:
        recs += RECOMMENDATIONS["exercise"]
    if input_data.get("dietary_habits_Unhealthy", 0) == 1:
        recs += RECOMMENDATIONS["diet"]

    # Deduplicate while preserving order
    seen, recs_dedup = set(), []
    for r in recs:
        if r not in seen:
            seen.add(r)
            recs_dedup.append(r)

    explanation = opening + factor_text
    if not factor_text:
        explanation += "Không có yếu tố nào nổi bật làm tăng nguy cơ trầm cảm ở mức độ đáng lo ngại."

    return explanation, recs_dedup[:6]
