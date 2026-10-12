DASDG 统一软件 1.1.0（正式发行材料，公开发布另行完成）

软件包括训练、预测、完整状态续跑、证据核验及冻结实验复现源码。
服务器短验收只用合成数据，不是论文性能实验。
公开旧仓库、旧验收包与正式实验结果均未修改。

通用入口：from dasdg.api import Config, fit, predict
命令入口：dasdg train / predict / evidence / status / dgcrn-prepare
训练文件：flow[时间,节点]完整有限标签；observed同形布尔掩码，True表示已观测；
split为训练、验证和总长度三个整数边界。不能训练目标标签也缺失的数据。
预测只读取历史输入，返回原量纲的[预测步长,节点]。
断点续跑按epoch边界恢复；配置、数据、源码或环境变化会拒绝续跑。
通用数据接口采用训练标签float64归一化，与冻结PEMS协议区分。

frozen目录保存原实验源码和配置；不包含数据、权重或DGCRN作者源码。
DGCRN上游LICENSE和README许可标识冲突尚未澄清。
tools/acquire_frozen_sources.py只供用户单独获取固定版本；不在验收中自动执行。
上游来源、固定提交、校验值和署名需保留。
英文README.txt给出API、复现入口、支持范围和限制。
外部DGCRN原源码还导入pandas，单独复现实验时需在相应环境安装。
保留的package_manifest.json及evidence check-engine针对原始完整实验归档；
本精简源码包有意不包含其中的上游源码等文件，因此用SHA256SUMS.txt和
source_provenance.json核验本包，不能用旧归档清单判定本包完整性。

1.1.0rc1已通过20项CPU核心与17项统一软件/GPU验收；1.1.0训练预测源码
逐字节保留，版本元数据和安装验收另有记录，见release_provenance.json与evidence。
嵌入的dasdg_reuse模块保留0.1.0rc1标识以保持检查点源码身份；整套发行版为1.1.0。
archival/v1.0.2保存旧版资料，其中旧基线预算不等于返修公平比较协议。
本次无需重复已完成的正式训练。需要自行验证安装时可运行短验收脚本。
概念DOI：10.5281/zenodo.22070217，覆盖各版本。
旧版v1.0.2的DOI：10.5281/zenodo.22074995；新版本DOI由Zenodo另行分配。
