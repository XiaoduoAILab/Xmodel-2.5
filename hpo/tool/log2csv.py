# save as log2csv.py
import re
import pandas as pd

log = open("run.log").read()          # 把整段日志保存成 run.log

pattern = re.compile(
    r"Trial (\d+) finished with value: ([\d\.inf]+) and parameters: "
    r"\{'learning_rate': ([\d\.]+), 'use_mup': (True|False), 'hidden_size': (\d+)\}"
)

rows = []
for m in pattern.finditer(log):
    rows.append({
        "trial_id": int(m.group(1)),
        "value": float(m.group(2)) if m.group(2)!="inf" else float("inf"),
        "learning_rate": float(m.group(3)),
        "use_mup": m.group(4) == "True",
        "hidden_size": int(m.group(5)),
    })

df = pd.DataFrame(rows).sort_values("trial_id")
df.to_csv("optuna_finished_trials.csv", index=False)
print("已生成 optuna_finished_trials.csv，共", len(df), "条记录")