# 实验三：小型LLM后训练与评估 (RLVR/GRPO)

## 实验目标与项目介绍

本实验旨在帮助学生掌握大语言模型（LLM）后训练（Post-training）中的强化学习（RL）技术和流程。通过本实验，学生将学习如何基于 **GRPO (Group Relative Policy Optimization)** 算法，对基座模型进行对齐训练，使其具备类似 DeepSeek-R1-Zero 的推理能力（Aha-Moment）。

实验内容涵盖了环境搭建、权重转换、数据预处理、Reward（奖励）规则设计、分布式强化学习训练执行以及模型能力评估。

本项目的实验代码基于 [MindSpeed-RL](https://gitcode.com/Ascend/MindSpeed-RL) 构建，复现了 DeepSeek-R1-Zero 在 Qwen2.5-7B 上的数学推理增强效果。

## 实验依赖与环境配置

本项目的基本依赖情况如下表所示（基于商分 2.1.0 版本配套）：

| 加速卡型号 | 驱动和CANN版本 | Python版本 | 主要Python包依赖 | MindSpeed-RL版本 |
|------------|----------------|------------|------------------|------------------|
| 昇腾910B   | Ascend HDK 25.3.0，**CANN 8.2.RC1** | Python 3.10  | torch 2.5.1，torch-npu 2.5.1，ray 2.42.1 | 分支 2.1.0 |

### 1. 容器环境准备（重要）

为了避免破坏服务器上现有的 `cann:8.2.rc1` 基础镜像，请**务必**按照以下步骤克隆一个新的实验镜像并启动容器：

```bash
# 1. 基于现有的 8.2.rc1 容器/镜像创建一个新镜像 (假设原镜像ID为 6d1a2236a49b)
# 注意：请将 'my_lab_image:v1' 替换为你自己的名字，如 'zhangsan_rl_lab:v1'
ContainerName='my_lab_image:v1'
docker commit 6d1a2236a49b $ContainerName

# 2. 启动新的实验容器
# 修改/home/your_name/workspace:/workspace  请挂载你的工作目录到 /workspace 修改my_rl_experiment为自己的实验名
# 例如我的工作目录放在/data1:/data1 下，则
My_RL_Experiment='GRPO'
WorkSpace='/data1:/data1'
docker run -itd \
  --name ${My_RL_Experiment} \
  --net=host \
  --shm-size=500g \
  --privileged \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${WorkSpace} \
  $ContainerName \
  /bin/bash

# 3. 进入容器
docker exec -it $My_RL_Experiment bash
```

### 2. 软件安装与源码准备

进入容器后，请参考
[安装指南](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97)配置 MindSpeed-RL 2.1.0 环境：
CANN已安装cann-8.2.rc1
部分安装代码如下

### vllm及相关依赖安装：
（注：环境中需要安装git，因为vllm的安装过程依赖git）
```shell
# pydantic高版本包会产生冲突，指定版本安装
pip install pydantic==2.12.0
git clone -b releases/v0.9.1 https://github.com/vllm-project/vllm.git
cd vllm
git checkout b6553be1bc75f046b00046a4ad7576364d03c835
VLLM_TARGET_DEVICE=empty pip install .
cd ..
```

### vllm_ascend安装
```shell
git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 8c7bc45
pip install -r requirements.txt
pip install -e .
cd ..
```

### ray安装

```shell
pip install ray==2.42.1
```
```shell
# ray 生成的日志文件夹权限修改
# 此处针对 ray==2.42.1 实现
RAY_PATH=$(python -c "import ray; print(ray.__file__)")
UTILS_PATH=$(dirname "$RAY_PATH")"/_private/utils.py"
sed -i 's/os.chmod(\(.*\), 0o0777)/os.chmod(\1, 0o0750)/g' "$UTILS_PATH"
```

### PyTorch框架安装
（注：[PyTorch框架和torch_npu插件安装教程](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)；可从[PyTorch-Ascend官方代码仓](https://gitcode.com/Ascend/pytorch/releases)获取PyTorch各个版本对应的torch_npu的whl包）
```shell
# 安装torch和torch_npu
#pip install torch-2.5.1-cp310-cp310-*.whl
#pip install torch_npu-2.5.1.*.manylinux2014_*.whl
```
#### 下载torch软件包和安装命令
```
yum install wget
wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

#### 下载torch_npu软件包和安装命令
```
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0-pytorch2.5.1/torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```
<!-- https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0050.html -->
### apex for Ascend 构建参考 https://gitcode.com/Ascend/apex
对于本实验环境
```
| CANN版本 | PyTorch版本 | Ascend Extension for PyTorch版本 | Python版本 | Apex 版本或代码分支 |
|:--------|:--------- |:-------------------------------|:--------|:------------------|
| 8.2.RC1  | 2.5.1      | v2.5.1-7.1.0                   | Python3.10x | master |

# 安装依赖
yum install -y patch libjpeg-turbo-devel dos2unix openblas git 
yum install -y gcc==7.3.0 cmake==3.12.0

# 请确保已安装PyTorch框架且setuptools版本小于等于65.7.0，若版本不符合条件，可使用以下命令安装
pip install setuptools==65.7.0
# 获取昇腾适配的Apex-patch源码
git clone -b master https://gitcode.com/Ascend/apex.git
cd apex/
# 执行
bash scripts/build.sh --python=3.10
# pip install apex-0.1.dev*.whl
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl  # version为Python版本和CPU架构
```




<!-- ```bash
# 1. 安装基础依赖
pip install ray==2.42.1
pip install pydantic==2.12.0
# 修正 Ray 权限
RAY_PATH=$(python -c "import ray; print(ray.__file__)")
UTILS_PATH=$(dirname "$RAY_PATH")"/_private/utils.py"
sed -i 's/os.chmod(\(.*\), 0o0777)/os.chmod(\1, 0o0750)/g' "$UTILS_PATH" -->
### 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
### OpenEuler 操作系统

执行如下命令重操作系统源安装jemalloc
```shell
yum install jemalloc
```
如果上述方法无法正常安装，可以通过源码编译安装
前往jemalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/
```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
#根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2,可通过以下命令来设置环境变量(可通过 find /usr -name libjemalloc.so.2 确认文件是否存在)
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2
```



# 2. 准备源码 (使用 2.1.0 配套版本)
```shell
git clone https://gitcode.com/Ascend/MindSpeed-RL.git

git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 89f4632d2cbb1e583a69b2bf3a08d75222f1173d  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../MindSpeed-RL/
cd ..

# Megatron从github下载，请确保网络能访问
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-RL/
cd ..

git clone https://gitcode.com/Ascend/MindSpeed-LLM.git -b 2.1.0
cd MindSpeed-LLM
git checkout 887c2d8682021befd675bb03965dbdee4de24516
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.9.3 --no-deps 
```
<!-- git clone -b 2.1.0 https://gitcode.com/Ascend/MindSpeed-RL.git

# 准备 MindSpeed-LLM (依赖)
git clone -b 2.1.0 https://gitcode.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
# 安装依赖
pip install -r requirements.txt
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

# 准备 Megatron-LM (Core 0.8.0)
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

# 3. 安装 MindSpeed-RL
cd MindSpeed-RL
pip install -r requirements.txt
``` -->

## 实验设计与指导

### 实验设定与流程

我们对小型LLM的强化学习后训练任务进行了如下设定：
- **模型设定**：使用 `Qwen2.5-7B` 作为 Base 模型。该模型指令遵从度高，适合作为 R1-Zero 流程的起点。
- **算法设定**：采用 **GRPO (Group Relative Policy Optimization)**。与 PPO 不同，GRPO 不需要额外的 Value Network (Critic)，通过分组采样和组内优势估计来优化策略，节省显存并提升训练效率。
- **数据集**：使用 `DeepScaler-Preview-Dataset` (40K条数学推理数据)。
- **研究性任务**：观察训练过程中的 "Aha-Moment"（顿悟时刻），即模型开始自发生成 `<think>` 标签并进行自我修正的过程。

<!-- 请参考下图工作流合理分配小组工作，安排实验进度：

![实验工作流](../../sources/images/r1_zero/r1_zero_roadmap.png) -->

### 数据集准备与预处理

1.  **下载数据**：下载 DeepScaler 数据集（Parquet格式）。
2.  **配置模板**：在 `configs/datasets/deepscaler.yaml` 中确认数据映射。
3.  **执行预处理**：将数据转换为训练所需的格式，并添加 R1 风格的 Prompt 模板。

```bash
# 在 MindSpeed-RL 目录下执行
bash examples/data/preprocess_data.sh deepscaler
```
*注：Prompt 模板会自动包裹 `<think>` 和 `<answer>` 标签引导模型输出。*

### 模型权重转换

MindSpeed-RL 基于 Megatron 架构，需要将 HuggingFace 格式的权重转换为 Megatron 格式。

```bash
# 参考 examples/mcore/qwen25/convert_ckpt_hf2mcore.sh
# 修改脚本中的 LOAD_CHECKPOINT_PATH 为你下载的 Qwen2.5-7B 路径
# 修改 SAVE_CHECKPOINT_PATH 为转换后的保存路径
bash examples/mcore/qwen25/convert_ckpt_hf2mcore.sh
```

### 模型后训练 (GRPO)

本实验使用基于规则的奖励模型（Rule-based Reward），不训练独立的 Reward Model。

1.  **启动 Ray 集群**：
    GRPO 训练依赖 Ray 进行分布式调度。
    ```bash
    # 在主节点启动 Ray head
    export MASTER_ADDR=localhost
    ray start --head --port 6344 --dashboard-host=$MASTER_ADDR --dashboard-port=8260
    ```

2.  **配置训练参数**：
    查看 `configs/grpo_qwen25_7b_A3.yaml`。关键参数说明：
    - `num_gpus`: 8 (使用单机8卡)
    - `kl_coeff`: KL 散度系数，控制模型不偏离基座太远。
    - `num_generations`: 组采样个数 (G)，通常设为 4 或 8。

3.  **启动训练**：
    ```bash
    # 设置环境变量
    export HCCL_CONNECT_TIMEOUT=1800
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    
    # 启动训练脚本
    bash examples/grpo/grpo_trainer_qwen25_7b.sh
    ```

### 模型评估

训练完成后，需要将 Megatron 权重转回 HuggingFace 格式，并使用数学评测集（如 MATH-500）进行评估。

1.  **权重反转**：使用 `convert_ckpt_mcore2hf.sh` 脚本。
2.  **推理评测**：使用 vLLM 或简单的推理脚本加载模型，测试其在数学问题上的 Pass@1 准确率。

### 消融实验 (选做)

针对 GRPO 训练过程，选择以下任一维度进行消融实验：
- **Reward 设计**：修改 `mindspeed_rl/reward/reward_rules.py`，调整格式奖励（Format Reward）和答案准确性奖励（Accuracy Reward）的权重，观察对收敛速度的影响。
- **采样数量 (G)**：调整 GRPO 的 `num_generations` (例如从 4 改为 8)，分析显存占用与训练效果的权衡。
- **迭代步数**：对比 200 iter 和 400 iter 的模型在 MATH-500 上的性能差异。

## 实践作业提交内容

1.  **环境截图**：Docker 容器启动成功及 `pip list` 包含 mindspeed-rl 的截图。
2.  **训练日志**：
    - 提供训练过程中的 Loss 曲线图（TensorBoard 截图或 Log 数据）。
    - 提供 Reward 变化曲线图（证明模型学到了规则）。
3.  **Aha-Moment 样例**：
    - 截取一个训练后的模型输出 Case，展示其 `<think>` 标签内的思考过程（特别是自我修正或长链推理的部分）。
4.  **实验报告**：
    - 记录实验步骤、遇到的问题及解决方案。
    - 分析消融实验的结果。