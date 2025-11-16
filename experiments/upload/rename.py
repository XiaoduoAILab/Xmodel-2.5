import os
import shutil

src_folder = "/data2/liuyang/i_line_ckpt/i_line_s1_fp8_0921/"
dst_folder = "/data2/liuyang/Xmodel-2.5-history/"

for sub in os.listdir(src_folder):
    iter_num = sub.split("_")[1]
    sub_path = os.path.join(src_folder, sub)
    # print(sub_path)
    src_path = os.path.join(sub_path, "pytorch_model.bin")
    dst_path = os.path.join(dst_folder, f"pytorch_model.{iter_num}")
    print(f'src_path: {src_path}')
    print(f'dst_path: {dst_path}')
    shutil.copy2(src_path, dst_path) 