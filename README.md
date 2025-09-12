1. github下载llava仓库
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
2. 下载依赖（需要隔离环境）
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
3. Hugging face下载llava模型
1. 下载hfd：
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
2. 安装aria2：
sudo apt update
sudo apt install aria2
3. 下载llava模型
#设置环境变量：
export HF_ENDPOINT=https://hf-mirror.com #下载模型报错重新把这句话复制一次
#创建储存文件夹:
mkdir -p <path>
cd <path>
#下载模型：
./hfd.sh liuhaotian/llava-v1.5-7b --local-dir <path>#下载模型
#下载数据集（以scienceQA为例）
./hfd.sh derek-thomas/ScienceQA --dataset --local-dir <path>
4. 运行：# Usage: 
 # bash run.sh docvqa test 4 "" /root/xxx
 # bash run.sh docvqa test 4 50   # 只跑 50 条