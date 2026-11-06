ICA 伪迹剔除管线（特征分类器版）
用随机森林对每个 ICA 成分的 12 个特征做分类，自动剔除伪迹成分。
本仓库只含代码和标注，不含原始 EEG 数据。

一、这个仓库里有什么
text
experiments/ica_ablation/
├── apply_final.py        # 总开关：训练 / 应用
├── ablation_lib.py       # 预处理、ICA、特征、ICLabel、数据路径
├── run_stage2.py         # 剔除 IC、算质量指标
├── labels/               # 人工标注（*_ses-*.json）
└── results/              # 训练产物 final_rf.joblib（默认不提交）
apply_final.py 只是总控，真正的预处理逻辑在 ablation_lib.py 里。

二、装环境
在项目文件夹下执行：

bash
pip install -r requirements.txt
建议用干净虚拟环境，别和你其他项目混在一起。

三、数据怎么放（最重要，先看这段）
本仓库不含原始 EEG 数据，需要你自己准备，并按下面的结构放好。
原始数据是 BIDS 格式的 EDF，根目录由 ablation_lib.py 顶部的 ROOT 决定：

python
ROOT = Path(r"D:\Downloads\BCI preprocessing")
BIDS_ROOT = ROOT / "nm000290"
拿到代码后，第一件事是把 ROOT 改成你本机数据所在的文件夹。 只改这一行，
BIDS_ROOT、缓存、输出、标注目录都会自动跟着变，不用改别处。

目录结构必须长这样（以 P01、ses-1 为例）：

text
<你的 ROOT>/
└── nm000290/
    └── sub-P01/
        └── ses-1/
            └── eeg/
                ├── sub-P01_ses-1_task-OnlineRawGraz_run-01_eeg.edf
                └── sub-P01_ses-1_task-OnlineRawGraz_run-01_events.tsv
命名规则（代码按这个规则找文件，对不上就报“找不到”）：

被试目录：sub-P01 … sub-P10，和 SUBJECTS 列表一一对应

session 目录：ses-1（本项目所有被试都是 1）

EDF 文件名：sub-<被试>_ses-1_task-OnlineRawGraz_run-<编号>_eeg.edf

同一被试同一 session 有多个 run 时，代码只取 run 编号最小的那个

同目录下的 _events.tsv 与 EDF 同名，只换后缀

被试列表写死在 ablation_lib.py 的 SUBJECTS 里（默认 P01–P10）。
数据不是这 10 个就改那一行。

ROOT 同时决定输出位置：干净数据、缓存、日志都会写在
<你的 ROOT>/experiments/ica_ablation/ 下。
原始 BIDS 文件夹只读不写，不会被改动。

四、第一次运行（推荐：本地重训）
不要直接加载别人机器上训好的 .joblib。 scikit-learn 跨版本不保证兼容，
典型报错是 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'。

在自己机器上重训一次，生成适配当前环境的模型：

bash
python -I apply_final.py --retrain
五、只跑部分被试
bash
python -I apply_final.py --subjects P01 P03
六、输出在哪
跑完都在 output/ 下：

<key>_desc-clean_eeg.fif：清洗后的数据

<key>_rejection.csv：每个 IC 的伪迹概率、是否剔除、ICLabel 对照

summary.json：所有记录的剔除数汇总

七、换数据时的注意事项
只换被试、同设备同流程：可以直接跑。先拿 1–2 条抽检，看 summary.json
里剔除数是否在 5–30 这个量级，正常再跑全部。

改了预处理参数（滤波、ICA 成分数等）：要先手动删掉对应缓存
（raw_preproc.fif、ica.fif、ic_features.csv），
否则会沿用旧缓存，你以为是新参数，其实是旧结果。

标注标准变了，或抽检发现系统性偏差：--retrain 重训。

八、方法边界（请务必知道）
分类器是在本数据集的人工标注上训练的。同一批数据上表现接近人工，
换新被试的泛化水平约 F1 ≈ 0.88（留一验证估计）。
换数据请先抽检，必要时重训。
