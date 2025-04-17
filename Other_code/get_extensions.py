import os
import argparse


def get_extensions(path):
    """用来获取目录以及子目录的扩展名，存放在集合中"""
    extensions = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext:
                extensions.add(ext.lower())
    return extensions

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'E:\Dissertation_Project\Road_Area_Recognition', help='输入一个路径')
opt = parser.parse_args()


if __name__ == '__main__':
    extensions = get_extensions(opt.path)
    print(extensions)
    
    

# os.walk()实现递归的原理：深度优先
# import os

# def walk(top, topdown=True):
#     # 生成器函数，用于模拟遍历目录树
#     try:
#         # 获取目录中的文件和子目录
#         names = os.listdir(top)
#     except OSError:
#         # 如果无法访问目录，返回一个空的生成器
#         return

#     dirs, nondirs = [], []
#     for name in names:
#         # 根据路径判断是文件还是目录
#         full_name = os.path.join(top, name)
#         if os.path.isdir(full_name):
#             dirs.append(name)  # 是目录，加入 dirs 列表
#         else:
#             nondirs.append(name)  # 是文件，加入 nondirs 列表

#     # 生成当前目录信息
#     yield top, dirs, nondirs

#     if topdown:
#         # 如果是从顶向下遍历，先返回当前目录，再遍历子目录
#         for dirname in dirs:
#             # 对子目录递归调用 walk
#             new_dir = os.path.join(top, dirname)
#             for x in walk(new_dir, topdown):
#                 yield x
#     else:
#         # 如果是从底向上遍历，则遍历子目录后再返回当前目录
#         for dirname in dirs:
#             new_dir = os.path.join(top, dirname)
#             for x in walk(new_dir, topdown=False):
#                 yield x
