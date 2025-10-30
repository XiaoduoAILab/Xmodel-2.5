#!/bin/bash

# 设置目标目录
if [ -z "$1" ]; then
  TARGET_DIR="/data13/datasets/pretrain/ultrafineweb-en_json/"
  echo "Using default directory: $TARGET_DIR"
else
  TARGET_DIR="$1"
fi

# 检查目标目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory $TARGET_DIR does not exist."
  exit 1
fi

# 删除文件名中包含单引号、星号或大括号的文件
echo "Searching for files with special characters (', *, {) in their names in $TARGET_DIR..."
find "$TARGET_DIR" -type f \( -name "*'*" -o -name "*\**" -o -name "*{*" \) -exec rm -f {} \;

echo "Files with special characters have been deleted."
