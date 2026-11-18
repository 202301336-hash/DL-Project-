import os
import json
import pandas as pd

os.makedirs("Final_Team_Delivery", exist_ok=True)

# 1. قراءة نتائج Model 3 (AST) من ملف test_summary.csv الجاهز
ast_summary_path = os.path.join("Member 3_ast", "experiments", "ast_probe", "final_test", "test_summary.csv")
ast_df = pd.read_csv(ast_summary_path) if os.path.exists(ast_summary_path) else None

# 2. قراءة نتائج Model 2 (2D-CNN) من ملف final_experiment_summary.json
m2_summary_path = os.path.join("Member 2_classic_dl", "results", "final_experiment_summary.json")
m2_data = json.load(open(m2_summary_path, "r", encoding="utf-8")) if os.path.exists(m2_summary_path) else {}

# 3. قراءة نتائج Model 1 (Advanced) من مجلد results
m1_files = [f for f in os.listdir("results") if f.endswith(".json") or f.endswith(".csv")] if os.path.exists("results") else []

print("\n" + "="*65)
print("📊 ملخص مقارنة الموديلات الثلاثة (للعرض على المعيد والتقرير)")
print("="*65)

if ast_df is not None:
    print("\n▶ نتائج Model 3 (AST - Test Set عبر 3 Seeds):")
    print(ast_df.to_string(index=False))

if m2_data:
    print("\n▶ نتائج Model 2 (Classic 2D-CNN):")
    print(json.dumps(m2_data, indent=2, ensure_ascii=False))

if m1_files:
    print("\n▶ ملفات نتائج Model 1 (Advanced) في مجلد results/:")
    for f in m1_files:
        print("  -", f)

# حفظ نسخة مجمعة في مجلد التسليم النهائي
if ast_df is not None:
    ast_df.to_csv(os.path.join("Final_Team_Delivery", "Model3_AST_Test_Comparison.csv"), index=False)

with open(os.path.join("Final_Team_Delivery", "Model2_2DCNN_Comparison.json"), "w", encoding="utf-8") as f:
    json.dump(m2_data, f, indent=4, ensure_ascii=False)

print("\n✅ تم تجميع ملفات المقارنة في مجلد: Final_Team_Delivery/")