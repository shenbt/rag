#!/bin/bash
# 保存为 download_models.sh，然后在 AutoDL 执行：bash download_models.sh

# 切换到 AutoDL 默认存储目录
cd /root/autodl-tmp

echo "🔄 开始下载 HuggingFace 模型 (使用 ghproxy 国内镜像加速)..."

# 1. sentence-transformers/all-MiniLM-L6-v2
if [ ! -d "all-MiniLM-L6-v2" ]; then
    git clone https://mirror.ghproxy.com/https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
else
    echo "✅ all-MiniLM-L6-v2 已存在，跳过下载"
fi

# 2. sentence-transformers/paraphrase-MiniLM-L3-v2
if [ ! -d "paraphrase-MiniLM-L3-v2" ]; then
    git clone https://mirror.ghproxy.com/https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2
else
    echo "✅ paraphrase-MiniLM-L3-v2 已存在，跳过下载"
fi

# 3. cross-encoder/ms-marco-MiniLM-L-6-v2
if [ ! -d "ms-marco-MiniLM-L-6-v2" ]; then
    git clone https://mirror.ghproxy.com/https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
else
    echo "✅ ms-marco-MiniLM-L-6-v2 已存在，跳过下载"
fi

echo "🎉 模型下载完成！全部保存在 /root/autodl-tmp/"
