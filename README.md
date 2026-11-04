ICA 伪迹剔除管线（特征分类器版）
用随机森林对每个 ICA 成分的 12 个特征做分类，自动剔除伪迹成分。
本仓库只含代码和标注，不含原始 EEG 数据。

目录结构
text
experiments/ica_ablation/
├── apply_final.py        # 总开关：训练 / 应用
├── ablation_lib.py       # 预处理、ICA、特征、ICLabel
├── run_stage2.py         # 剔除 IC、算质量指标
├── labels/               # 人工标注（*_ses-*.json）
└── results/              # 训练产物 final_rf.joblib（默认不提交）
数据路径由 ablation_lib.py 里的 al.SUBJECTS / al.cache_dir() / al.LABEL_DIR 决定，
请按实际位置放置原始数据，保持相对结构一致。

环境
bash
pip install -r requirements.txt
第一次运行（推荐：本地重训）
不要直接加载别人机器上训好的 .joblib——scikit-learn 跨版本不保证兼容
（典型报错：'DecisionTreeClassifier' object has no attribute 'monotonic_cst'）。

在自己机器上重训一次，生成适配当前环境的模型：

bash
python -I apply_final.py --retrain
只跑部分被试
bash
python -I apply_final.py --subjects P01 P03
输出
output/<key>_desc-clean_eeg.fif —— 清洗后的数据

output/<key>_rejection.csv —— 每个 IC 的伪迹概率、是否剔除、ICLabel 对照

output/summary.json —— 剔除数汇总

换数据时注意
只换被试、同设备同流程：可直接跑，先拿 1–2 条抽检，看 summary.json 剔除数是否在 5–30 量级。

改了预处理参数（滤波、ICA 成分数等）：先手动删除对应缓存
（raw_preproc.fif、ica.fif、ic_features.csv），否则会沿用旧缓存。

标注标准变了 / 抽检发现系统性偏差：--retrain 重训。

方法边界
分类器在本数据集的人工标注上训练。同一批数据上表现接近人工，
换新被试的泛化水平约 F1 ≈ 0.88（留一验证估计）。换数据请先抽检，必要时重训。
