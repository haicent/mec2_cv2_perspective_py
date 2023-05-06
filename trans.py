# 核心思想：利用cv2提供的仿射变换函数，来完成投影类坐标转换。
import cv2
import numpy as np


def img_trans():
    '''
    这个函数，是用图片来单独示例坐标转换的效果的。
    '''
    img = cv2.imread("haha.png")
    h, w, c = img.shape
    src_list = [[207, 226], [119, 882], [1234, 853], [1099, 72]]
    # src_list = [[420, 421], [223, 656], [852, 781], [842, 477]]
    # src_list = [[658, 343], [353, 941], [914, 940], [945, 338]]
    for i, pt in enumerate(src_list):
        cv2.circle(img, pt, 5, (0, 255, 255), -1)
        cv2.putText(img, str(i + 1), (pt[0] + 5, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    pts1 = np.float32(src_list)

    # pts2 = np.float32([[0, 0],
    #                    [161, 0],
    #                    [161, 161],
    #                    [0, 161]])
    # pts2 = np.float32([[811.5, 1000-573.86],
    #                    [884.4, 1000-558.62],
    #                    [1000.6, 1000-598.24],
    #                    [823.6, 1000-607.32]])
    # pts2 = np.float32([[6601, 8537],
    #                    [7176, 8483],
    #                    [7193, 8555],
    #                    [6208, 8769]])
    # 为了正确的显示转换后的图像，把四个目标点的经纬度做了转换：截断了长度，把纬度做了翻转- 和平移1000.
    pts2 = np.float32([[600.85, 1000-537.02],
                       [1175.55, 1000-482.93],
                       [1192.63, 1000-554.63],
                       [207.79, 1000-769.26]])
    # pts2 = np.float32([[500, 900], [500, 1000], [900, 1150], [750, 900]])
    # 生成仿射变换矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 序列化矩阵，存入文件中
    np.save('tmp/matrix.npy', matrix)
    # 反序列化，从文件中取出
    b = np.load('tmp/matrix.npy')

    # 按照新的坐标系，把之前的图片进行转换（只显示四个坐标中间的部分，但是其他未显示的部分其实可以通过函数进行坐标转换）
    # result = cv2.warpPerspective(img, matrix, (h, w))
    result = cv2.warpPerspective(img, matrix, (1500, 1500))

    vedio_xy = np.array([[473,92],[473,93],[473,94],[474,92],[475,92]], dtype=np.float32).reshape(-1, 1, 2)
    real_xy = cv2.perspectiveTransform(vedio_xy, matrix)
    print('real_xy:',np.round(real_xy,6))
    cv2.imshow("Image", img)
    cv2.imshow("mec-仿射转换", result)
    cv2.waitKey()



def xy_trans():
    '''
    这个函数，是用来示例如何进行坐标转换的，考虑到了精读问题。
    :return:
    '''
    # 录像中的 道路四个角的坐标
    corner_video = np.float32([[207, 226], [119, 882], [1234, 853], [1099, 72]])
    # 真实世界中 对应道路四个角的坐标
    corner_real_list = [[121.47660085833333, 37.448537019444444],
                        [121.47717555277778, 37.448482925],
                        [121.477192625, 37.44855463611111],
                        [121.47620778611112, 37.44876926111111]]

    # corner_real_list = [[121.476601, 37.448537],
    #                     [121.477176, 37.448483],
    #                     [121.477193, 37.448555],
    #                     [121.476208, 37.448769]]

    x = np.float64(corner_real_list) * 100
    float_part = np.modf(x)[0]     # 取出小数部分
    int_part = np.modf(x)[1][0]    # 取出整数部分

    corner_real_f32 = np.float32(float_part)

    # 生成转换矩阵
    tras_matrix = cv2.getPerspectiveTransform(corner_video, corner_real_f32)

    # 视频中的平面上，指定点的坐标
    vedio_xy = np.array([[473,92]], dtype=np.float32).reshape(-1, 1, 2)

    # 视频中任意一个点，对应真实世界坐标系统的坐标
    # 小数部分
    f_part_real_xy = cv2.perspectiveTransform(vedio_xy, tras_matrix)

    # 变换为之前的量级
    real_car_xy = np.around((int_part + f_part_real_xy) * 0.01, decimals=9)

    print('视频中坐标：', vedio_xy)
    print('真实经纬度：', np.round(real_car_xy,6))  # 注意：print打印时，控制台只显示8位。但是其实，该经纬度的精度到第九位。如果需要打印9位真实值，可以加上下标[0][0][0] 如下所示
    print('真实经纬度：', real_car_xy[0][0][0], real_car_xy[0][0][1])



if __name__ == "__main__":
    img_trans()
    xy_trans()

    pass
