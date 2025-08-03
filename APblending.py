import numpy as np
from PIL import Image
import time


#请严格保持两图大小相同，目标物对应
#不同也能运行，但会发生错位
#当颜色差距过大是，可能效果不会很好（比如：红与蓝）

#使用方法
#将表图命名为f.png，里图命名为b.png（一定得是png，jpg重命名是不可以的）
#然后直接运行就可以了
#毕竟不会向量，也不知道怎么用gpu加速计算，所以算法局限，可能有点慢
#处理两个1M的图片大概要1min

def find_a(rgb1:tuple , rgb2:tuple,bright_background:tuple,black_ground:tuple):
    def lume(rgb):
        #计算图像参数，带权重
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    
    l1,l2,lb,lk = lume(rgb1),lume(rgb2),lume(bright_background),lume(black_ground)
    d = l2 - l1
    e = lb - lk
    x = 1 + np.true_divide(e*d,e**2)

    #约束范围
    x = min(1,max(1e-10,x))

    return x

def mix(r1:tuple , r2:tuple ,bright_background:tuple,black_ground:tuple, x):
    rgb1,rgb2,bg,dg = list(r1),list(r2),list(bright_background),list(black_ground)

    #防止除0
    rgb1[:] = [1e-10 if i == 0 else i for i in rgb1]
    rgb2[:] = [1e-10 if i == 0 else i for i in rgb2]
    bg[:] = [1e-10 if i == 0 else i for i in bg]
    dg[:] = [1e-10 if i == 0 else i for i in dg]

    x = max(x,1e-10)

    r =int(np.true_divide(((rgb1[0] + rgb2 [0]) - (bg[0] + dg[0])*(1-x)),(2*x)))
    g =int(np.true_divide(((rgb1[1] + rgb2 [1]) - (bg[1] + dg[1])*(1-x)),(2*x)))
    b =int(np.true_divide(((rgb1[2] + rgb2 [2]) - (bg[2] + dg[2])*(1-x)),(2*x)))
    x_1 =int (x * 255)

    r = max(0,min(255,r))
    g = max(0,min(255,g))
    b = max(0,min(255,b)) 
    
    return (r,g,b,x_1)

def pre_dark(img):
    #让图片降低50%亮度
    w, h =img.size
    back = Image.new("RGBA",(w,h),(0,0,0,255))
    mask = Image.new("RGBA",(w,h),(128,128,128,255))

    mixx=Image.composite(img,back,mask)

    return mixx

def pre_bringt(img):
    #让图片增白25%
    w, h =img.size
    back = Image.new("RGBA",(w,h),(255,255,255,255))
    mask = Image.new("RGBA",(w,h),(64,64,64,255))

    mixx=Image.composite(img,back,mask)

    return mixx

def main():
    #背景色，这里定义为不是很纯的黑和白
    #但这样结果会出现残影
    #不想要残影可以直接设置为（255，255，255），（0，0，0）
    bright = (255,255,255) #（255，255，255）
    black = (00,0,0) #（0，0，0）

    img_f,img_b = Image.open("f.png"),Image.open("b.png")

    #图像预处理，不想可以自己注释掉
    img_b = pre_dark(img_b)#变暗
    img_f = pre_bringt(img_f)#增亮

    pix_f,pix_b = img_f.load(),img_b.load() 

    w_f , h_f =img_f.size
    w_b , h_b =img_b.size
    #防止超出边界，会截断图像
    w,h = min(w_f,w_b),min(h_b,h_f)

    img = Image.new("RGBA",(w,h))
    pix = img.load()


    #逐个填入像素，作者不会用向量
    #会用向量的可以用numpy加速
    for y in range(h):
        for x in range(w):
            a = find_a(pix_f[x,y],pix_b[x,y],bright,black)
            pix[x,y] = mix(pix_b[x,y],pix_f[x,y],bright,black,a)
            
    time_now = int(time.time())
    #保存
    img.save(f"Mix{time_now}.png","PNG")

if __name__ == "__main__":
    main()