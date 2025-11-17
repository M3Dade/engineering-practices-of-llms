# 总体环境配置指导

## NPU设备和驱动检查

登陆华为云服务器后，执行`npu-smi info`命令，检查NPU设备和驱动是否正常。该命令将得到如下输出：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.6                   Version: 23.0.6                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 114.0       48                0    / 0             |
| 0                         | 0000:C1:00.0  | 6           0    / 0          5092 / 65536         |
+===========================+===============+====================================================+
| 1     910B2               | OK            | 89.5        49                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3334 / 65536         |
+===========================+===============+====================================================+
| 2     910B2               | OK            | 87.5        48                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3334 / 65536         |
+===========================+===============+====================================================+
| 3     910B2               | OK            | 90.3        51                0    / 0             |
| 0                         | 0000:02:00.0  | 0           0    / 0          3335 / 65536         |
+===========================+===============+====================================================+
| 4     910B2               | OK            | 94.1        47                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          3334 / 65536         |
+===========================+===============+====================================================+
| 5     910B2               | OK            | 94.0        49                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          3335 / 65536         |
+===========================+===============+====================================================+
| 6     910B2               | OK            | 92.2        49                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          3335 / 65536         |
+===========================+===============+====================================================+
| 7     910B2               | OK            | 91.2        50                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          3336 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 769280        | python                   | 1803                    |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 2                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 3                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 4                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 6                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
```

其中`HBM-Usage(MB)`一栏显示了每个NPU的显存占用情况，下方的进程列表则显示了当前系统中正在使用NPU的进程信息。请熟悉该信息界面，在具体实验实操过程中需通过该命令频繁检查NPU设备状态，以确保实验进程正常运行且显存使用情况符合预期。

执行`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`命令，检查CANN工具包版本，确保输出包含如下内容（注意确认`8.0.RC3`）：

```
# version: 1.0
runtime_running_version=[7.5.0.1.129:8.0.RC3]
compiler_running_version=[7.5.0.1.129:8.0.RC3]
opp_running_version=[7.5.0.1.129:8.0.RC3]
toolkit_running_version=[7.5.0.1.129:8.0.RC3]
aoe_running_version=[7.5.0.1.129:8.0.RC3]
ncs_running_version=[7.5.0.1.129:8.0.RC3]
hccl_running_version=[7.5.0.1.129:8.0.RC3]
opp_kernel_running_version=[7.5.0.1.129:8.0.RC3]
```

若上述信息打印异常，请联系管理员协助处理。

## 磁盘环境检查与配置

执行`df -h`命令，检查当前系统中是否存在大存储量（14TB 或 21TB）的磁盘分区（如`/data`分区），以确保后续实验数据和模型文件的存储需求。

若不存在大存储量磁盘分区，执行`lsblk`命令，检查是否有未挂载的磁盘设备。例如有三块7TB的磁盘设备`/dev/vdb`、`/dev/vdc`和`/dev/vdd`，则可执行以下命令将其合并挂载到`/data`目录下：

```bash
sudo mkfs.ext4 /dev/vdb
sudo mkfs.ext4 /dev/vdc
sudo mkfs.ext4 /dev/vdd 
sudo mkdir /data
sudo mount /dev/vdb /data
sudo mount /dev/vdc /data
sudo mount /dev/vdd /data
df -h  # 检查挂载成功
```

假设存在名为`/data`的大存储量磁盘分区，在该分区下创建实验所需的常用文件夹：

```bash
mkdir -p /data/models         # 用于存放下载下来的开源模型
mkdir -p /data/datasets       # 用于存放下载下来的数据集
mkdir -p /data/checkpoints    # 用于存放实验过程中保存的模型检查点
```

注意本项目中的后续实验均使用上述文件夹作为公共的数据和模型存储路径，请确保该目录具备足够的存储空间。

## screen环境安装与使用教程

screen工具用于在远程服务器上创建可分离的终端会话，确保实验过程中即使网络连接中断，实验任务仍能继续运行（如果你更擅长使用`tmux`工具，也可以自行安装和使用）。请执行以下命令安装screen工具：

```bash
sudo yum install screen -y
```

安装完成后，可以通过以下命令创建一个新的screen会话：

```bash
screen -S session_name
```

在创建的screen会话中，可以使用`Ctrl+A`接`D`键来退出screen会话，会话中的进程将继续运行。要重新连接到之前的screen会话，可以使用以下命令：

```bash
screen -rD session_name
```

可以开启多个screen会话，使用以下命令查看当前所有的screen会话：

```bash
screen -ls
```

## obsutil工具的安装与配置

obsutil工具用于在华为云对象存储服务（OBS）和服务器之间进行数据传输，课程项目组已经提前在OBS上传了实践课程所需的大型数据集和模型文件，大家需要使用obsutil工具将这些数据集和模型下载到自己的服务器上。各实验所需的数据集和模型下载请参考对应实验文件夹的`README.md`，本小节将介绍obsutil工具的安装和配置方法。

执行以下命令下载并安装obsutil工具：

```bash
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
tar -zxvf obsutil_linux_arm64.tar.gz
cd obsutil_linux_arm64_x.x.x  # x.x.x为具体版本号
sudo mv obsutil /usr/bin/
```

执行以下命令配置obsutil工具：

```bash
obsutil config -i=HPUAFQG6POXNNRG07RMQ -k=AB7vwTC466LVbGo1vTLL34tvbdIx3TWgXolG40Qj -e=https://obs.cn-southwest-2.myhuaweicloud.com
obsutil ls -s  # 检查是否配置成功，成功则会列出OBS桶列表
```

后续下载举例：

```bash
obsutil cp obs://hangdian/pretrain_data /data/datasets/ -r -f  # 从OBS下载数据集 pretrain_data 到服务器的/data/datasets/目录下
```

## Python环境配置

我们使用 miniconda 来管理 Python 环境，因为不同实验可能需要不同的 Python 包和版本，conda 的虚拟环境功能可以帮助我们更好地管理这些依赖，隔离环境，自由切换。执行以下命令安装 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
```

配置默认的 pip 源为清华大学镜像，加快包的下载速度：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

下面我们举例说明如何创建和使用 conda 虚拟环境。假设我们需要创建一个名为`llm_env`的虚拟环境，并安装所需的 Python 包：

```bash
conda create -n llm_env python=3.10 -y  # 创建名为 llm_env 的虚拟环境，指定 Python 版本为 3.10
```

安装完成后，可以激活该虚拟环境:

```bash
conda activate llm_env
```

在虚拟环境中，可以使用`pip install`命令安装所需的 Python 包，例如`transformers`、`sentencepiece`等。

```bash
pip install transformers sentencepiece
```

要退出当前虚拟环境，可以使用以下命令：

```bash
conda deactivate
```

## clone 本代码仓库

请执行以下命令将本代码仓库克隆到本地服务器：

```bash
git clone https://github.com/milvlg/LLM-Practice-Course.git
```