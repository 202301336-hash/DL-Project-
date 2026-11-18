import os
import json
import subprocess

HANDOFF_DIR = "Final_Team_Delivery"
os.makedirs(HANDOFF_DIR, exist_ok=True)

env_model1 = os.environ.copy()
extra_paths = os.pathsep.join([
    os.path.abspath("."),
    os.path.abspath("data"),
    os.path.abspath("src"),
    os.path.abspath(os.path.join("src", "models")),
    os.path.abspath("scripts")
])
env_model1["PYTHONPATH"] = extra_paths + os.pathsep + env_model1.get("PYTHONPATH", "")

tasks = [
    {
        "name": "Model_1_Advanced",
        "cwd": ".",
        "cmd": ["py", "-3.14", "train_advanced.py"],
        "env": env_model1
    },
    {
        "name": "Model_2_Classic_2DCNN",
        "cwd": "Member 2_classic_dl",
        "cmd": ["py", "-3.14", "run_classic_dl.py"],
        "env": None
    },
    {
        "name": "Model_3_AST",
        "cwd": "Member 3_ast",
        "cmd": ["py", "-3.14", "train_ast_probe.py", "--audio", "../audio"],
        "env": None
    }
]

execution_status = {}
print("🚀 بدء تشغيل جميع موديلات المشروع بالترتيب...\n")

for task in tasks:
    print(f"==========================================")
    print(f"▶ جاري تشغيل: {task['name']}")
    print(f"==========================================")
    try:
        subprocess.run(task["cmd"], cwd=task["cwd"], env=task["env"], check=True)
        execution_status[task["name"]] = "Success ✅"
    except Exception as e:
        print(f"❌ خطأ في {task['name']}: {e}")
        execution_status[task["name"]] = f"Failed: {str(e)}"

summary_file = os.path.join(HANDOFF_DIR, "ALL_MODELS_SUMMARY.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump({
        "project": "ESC-50 Audio Classification",
        "data_split": {"train": 1280, "val": 320, "test": 400},
        "models_status": execution_status,
        "where_to_find_results": {
            "Model_1_Advanced": "results/ & saved_models/",
            "Model_2_Classic_2DCNN": "Member 2_classic_dl/results/final_experiment_summary.json & Member 2_classic_dl/checkpoints/",
            "Model_3_AST": "Member 3_ast/results/ & Member 3_ast/experiments/"
        }
    }, f, indent=4, ensure_ascii=False)

print("\n🎉 تم الانتهاء! الملف الموحد جاهز في:", summary_file)