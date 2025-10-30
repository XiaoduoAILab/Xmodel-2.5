import os
import shutil

# folder = r'D:\Users\Administrator\models\Xmodel-2-history\s1'
# folder = "/data4/liuyang/Xmodel-2-history/s2"
folder = "/data1/liuyang/Xmodel-2-history/decay"

for sub in os.listdir(folder):
    iter_num = sub.split("-")[1]
    sub_path = os.path.join(folder, sub)
    # print(sub_path)
    src_path = os.path.join(sub_path, "pytorch_model.bin")
    dst_path = os.path.join(folder, f"pytorch_model.{iter_num}")
    print(f'src_path: {src_path}')
    print(f'dst_path: {dst_path}')
    shutil.copy2(src_path, dst_path) 