from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import numpy as np
from tqdm import tqdm
"""
将标签json文件转为可识别的可行驶区域标签图像
"""
#传入的变量:图形坐标，是否闭合，不透明度，颜色
#其实这个函数主要是设置好了Path(points,codes)
def poly2patch(poly2d,closed=False,alpha=1,color=None):
    moves = {'L':Path.LINETO,     #'L' 映射为 Path.LINETO，表示绘制一条直线。
             'C':Path.CURVE4}     #'C' 映射为 Path.CURVE4，表示绘制一个4次贝塞尔曲线。
    points = [p[:2] for p in poly2d]       #points是一个列表，存放要画的坐标
    codes = [moves[p[2]] for p in poly2d]  #存放要画的线条的类型
    codes[0] = Path.MOVETO                 #代表第一个点是起始点
    
    #如果要闭合的图,把第一个点加在最后,Path.CLOSEPOLY表示图形要闭合
    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)

    #返回值是使用mpatches.PathPatch创建的对象，它会将路径渲染成一个图形
    return mpatches.PathPatch(
        Path(points,codes),                         #构造一个Path对象 
        facecolor = color if closed else 'none',    #如果路径是闭合的，设置填充颜色
        edgecolor = color,                          #设置边界的颜色
        lw = 1 if closed else 2*1,
        alpha = alpha,                              #设置透明度
        antialiased = False,                        #禁用抗锯齿
        snap = True                                 #启用对路径点的自动对齐
    )

def get_area_v0(data):
    #返回一个列表，里面是满足条件的一系列词典
    return [o for o in data
            if 'poly2d' in o and o['category'].startswith('area')]

def draw_drivable(data,ax):
    plt.draw()
    #先筛选出包含"poly2d"和"category": "area的键的元素
    data = get_area_v0(data)
    for obj in data:
        if obj['category'] == 'area/drivable' or obj['category'] == 'area/alternative':
            color = (1,1,1)#白色
        else:
            if obj['category'] != 'area/drivable':
                pass
        alpha = 1.0 #定义不透明度
        poly2d = obj['poly2d'] #把坐标拿出来,例:[431.503844,220.094633,"L"]
        #添加图形补丁
        ax.add_patch(poly2patch(poly2d,closed=True,alpha=alpha,color=color))
        ax.axis('off')   #关闭坐标轴

def filter_pic(data):
    for obj in data:
        if obj['category'].startswith('area'):
            return True
        else:
            pass
    return False

def main(mode = 'val'):
    #文件路径
    # image_dir = r""
    label_dir = r"E:\BDD100K\{}\labels".format(mode)
    out_dir = r"E:\BDD100K\{}\labels_drivable".format(mode)
    #确保out_dir存在
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #读取json名，存在列表中
    label_list = os.listdir(label_dir)
    #完整循环一遍
    i = 0
    for label_json in tqdm(label_list):
        if i>50:
            break
        i+=1
        #加载并打开json
        label_json = os.path.join(label_dir,label_json)
        label_pd = json.load(open(label_json))
        #把json(词典复合结构)中的'objects'的值存起来'name'也存起来
        data = label_pd['frames'][0]['objects']
        img_name = label_pd['name']

        #画图
        dpi = 80
        w = 16
        h = 9
        image_width = 1280
        image_height = 720
        fig = plt.figure(figsize=(w,h),dpi=dpi)
        #设置坐标轴ax,但是不显示坐标轴
        ax = fig.add_axes([0.0,0.0,1.0,1.0],frameon=False)
        ax.set_xlim(0, image_width - 1)
        ax.set_ylim(0, image_height - 1)
        ax.invert_yaxis()#反转y轴，到左上角(图像坐标系)
        #添加图形补丁(矩形)
        ax.add_patch(poly2patch([[0,0,'L'],[0,image_height-1,'L'],
                                 [image_width-1,image_height-1,'L'],[image_width-1,0,'L']],
                                closed=True,alpha=1,color=(0,0,0)))

        #检查一下data中是否有需要的可行驶区域信息
        remain = filter_pic(data)
        if remain:
            draw_drivable(data,ax)#绘制可行驶区域
        #保存图片
        out_path = os.path.join(out_dir,img_name+'png')
        fig.savefig(out_path,dpi = None)
        plt.close()

if __name__ == '__main__':
    main(mode = 'train')
    # main(mode = 'val')