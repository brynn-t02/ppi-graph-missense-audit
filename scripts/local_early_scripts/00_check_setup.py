"""
脚本名：00_check_setup.py

功能：
    检查 v2 工程的目录、配置文件、数据文件是否全部可正常读取。

为什么做：
    这是所有后续训练与评估脚本的地基。
    如果这一步不通，后面所有脚本都会乱。

输入：
    - configs/main.yaml
    - data/processed/clinvar_train_cold_start.csv
    - data/processed/clinvar_test_with_AM_final.csv

输出：
    - 终端打印检查结果

是否主结果：
    是（工程初始化检查）
"""

from pathlib import Path
import pandas as pd
import yaml
import re

# ========= 1. 自动定位项目根目录 =========
ROOT = Path(__file__).resolve().parents[1]

# ========= 2. 读取配置 =========
config_path = ROOT / "configs" / "main.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

train_path = ROOT / cfg["paths"]["train_csv"]
test_path = ROOT / cfg["paths"]["test_csv"]
regex = cfg["filters"]["missense_exclude_regex"]

print("=" * 60)
print("🔍 V2 工程初始化检查")
print("=" * 60)
print(f"项目根目录: {ROOT}")
print(f"配置文件: {config_path}")
print(f"训练集路径: {train_path}")
print(f"测试集路径: {test_path}")
print(f"过滤正则: {regex}")

# ========= 3. 检查文件存在 =========
assert config_path.exists(), f"配置文件不存在: {config_path}"
assert train_path.exists(), f"训练集不存在: {train_path}"
assert test_path.exists(), f"测试集不存在: {test_path}"

print("\n✅ 文件存在检查通过")

# ========= 4. 读取数据 =========
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\n📊 原始数据规模")
print(f"训练集 shape: {train_df.shape}")
print(f"测试集 shape: {test_df.shape}")

# ========= 5. 应用测试集过滤 =========
test_df_filtered = test_df[~test_df["Name"].str.contains(regex, regex=True, na=False)].copy()

import re

print("\n=== [CHECK1-FINAL] 从 Name 中提取 HGVS.p ===")

# 更稳的提取方式：直接抓 Name 里的 p.xxx
hgvs_p = test_df["Name"].astype(str).str.extract(r"(p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}|p\.[A-Z][a-z]{2}\d+(?:Ter|\*))")[0]

print("Test total variants:", len(test_df))
print("Parsed HGVS.p available:", hgvs_p.notna().sum())

print("\n前10个成功提取的 HGVS.p 示例：")
print(hgvs_p.dropna().head(10).to_string(index=False))

# 严格 missense：p.Arg259Pro 这种
missense_mask = hgvs_p.str.match(r"^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", na=False)

# stop-gain：p.Gln232Ter 或 p.Gln232*
stop_mask = hgvs_p.str.match(r"^p\.[A-Z][a-z]{2}\d+(Ter|\*)$", na=False)

print("\n=== [CHECK2-FINAL] 变异类型统计 ===")
print("Missense-like:", missense_mask.sum())
print("Stop-gain-like:", stop_mask.sum())
print("Other / unparsable:", len(test_df) - missense_mask.sum() - stop_mask.sum())

print("\nMissense examples:")
print(hgvs_p[missense_mask].head(10).to_string(index=False))

print("\nStop-gain examples:")
print(hgvs_p[stop_mask].head(10).to_string(index=False))

# 额外检查：是否有 fs/del/dup/ins 这类明显非 missense
non_missense_regex = re.compile(r"fs|Ter|\*|del|dup|ins|=")
hit = test_df["Name"].astype(str).str.contains(non_missense_regex, regex=True, na=False)

print("\n=== [CHECK3-FINAL] Name 中疑似非 missense 关键词统计 ===")
print("Regex hit count:", hit.sum())
print("Regex hit rate:", round(hit.mean(), 4))

print("\n前10个命中 regex 的 Name：")
print(test_df.loc[hit, "Name"].head(10).to_string(index=False))


print("\n=== [DEBUG] Name examples ===")
print(test_df["Name"].head(5).to_string(index=False))
print("\nContains 'p.':", test_df["Name"].astype(str).str.contains(r"p\.", regex=True, na=False).sum())
print("Contains '(p.':", test_df["Name"].astype(str).str.contains(r"$p\.", regex=True, na=False).sum())


# 1) 从 Name 里抽取括号内的 HGVS.p，比如 (p.Arg330Met)
hgvs_p = test_df["Name"].astype(str).str.extract(r"$(p\.[^)]+)$")[0]

# 2) 严格判定 missense：p.Xxx123Yyy
missense_mask = hgvs_p.str.match(r"^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", na=False)




print("\n🧬 测试集过滤后规模")
print(f"过滤后测试集 shape: {test_df_filtered.shape}")
print(f"过滤后测试集变异数: {len(test_df_filtered)}")

# ========= 6. 打印关键列检查 =========
required_cols = ["Name", "Label", "Protein_ID"]
print("\n📌 关键列检查")
for col in required_cols:
    print(f"{col}: {'✅ 存在' if col in train_df.columns and col in test_df.columns else '❌ 缺失'}")

print("\n🏁 初始化检查完成")
print("=" * 60)
