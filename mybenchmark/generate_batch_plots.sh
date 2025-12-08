#!/bin/bash
# 生成给定 sequence_length 的不同 batch_size 图表的命令脚本

# 激活conda环境
conda activate trtllm

# 定义 sequence_length 值
SEQUENCE_LENGTHS=("128" "256" "512" "1024" "2048" "4096" "8192" "16384" "32768")

# 定义 TP size
TP_SIZE=2

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/TensorRT-LLM/mybenchmark/figures"

# 数据文件
DATA_FILE="/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/avg_bandwidth.csv"

echo "生成给定 sequence_length 的不同 batch_size 图表..."
echo "输出目录: $OUTPUT_DIR"
echo "数据文件: $DATA_FILE"
echo "TP size: $TP_SIZE"
echo ""

# 为每个 sequence_length 生成 batch_vs_bandwidth 图表
for SEQ_LEN in "${SEQUENCE_LENGTHS[@]}"; do
    echo "正在生成 sequence_length=$SEQ_LEN 的图表..."
    python plot_bandwidth_analysis.py \
        --data-file "$DATA_FILE" \
        --plot-type batch_vs_bandwidth \
        --fixed-value "$SEQ_LEN" \
        --tp-size "$TP_SIZE"
done

echo ""
echo "所有图表生成完成！"
echo "图表保存在: $OUTPUT_DIR"

# 列出生成的文件
echo ""
echo "生成的文件列表:"
ls -la "$OUTPUT_DIR"/batch_vs_bandwidth_*.png 2>/dev/null || echo "没有找到生成的图表文件"

# 提供批量查看命令
echo ""
echo "查看所有生成的图表:"
echo "eog $OUTPUT_DIR/batch_vs_bandwidth_*.png"