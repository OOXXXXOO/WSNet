def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50):
    '''Selective Search
    首先通过基于图的图像分割方法初始化原始区域，就是将图像分割成很多很多的小块
    然后我们使用贪心策略，计算每两个相邻的区域的相似度
    然后每次合并最相似的两块，直到最终只剩下一块完整的图片
    然后这其中每次产生的图像块包括合并的图像块我们都保存下来

    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size 候选区域大小，并不是边框的大小
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    # 图片分割 把候选区域标签合并到最后一个通道上 height x width x 4  每一个像素的值为[r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)


if img is None:
    return None, {}

    # 计算图像大小
imsize = img.shape[0] * img.shape[1]

# dict类型，键值为候选区域的标签   值为候选区域的信息，包括候选区域的边框，以及区域的大小，颜色直方图，纹理特征直方图等信息
R = _extract_regions(img)

# list类型 每一个元素都是邻居候选区域对(ri,rj)  (即两两相交的候选区域)
neighbours = _extract_neighbours(R)

# calculate initial similarities 初始化相似集合S = ϕ
S = {}
# 计算每一个邻居候选区域对的相似度s(ri,rj)
for (ai, ar), (bi, br) in neighbours:
    # S=S∪s(ri,rj)  ai表示候选区域ar的标签  比如当ai=1 bi=2 S[(1,2)就表示候选区域1和候选区域2的相似度
    S[(ai, bi)] = _calc_sim(ar, br, imsize)

# hierarchal search 层次搜索 直至相似度集合为空
while S != {}:

    # get highest similarity  获取相似度最高的两个候选区域  i,j表示候选区域标签
    i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]  # 按照相似度排序

    # merge corresponding regions  合并相似度最高的两个邻居候选区域 rt = ri∪rj ,R = R∪rt
    t = max(R.keys()) + 1.0
    R[t] = _merge_regions(R[i], R[j])

    # mark similarities for regions to be removed   获取需要删除的元素的键值
    key_to_delete = []
    for k, v in S.items():  # k表示邻居候选区域对(i,j)  v表示候选区域(i,j)表示相似度
        if (i in k) or (j in k):
            key_to_delete.append(k)

    # remove old similarities of related regions 移除候选区域ri对应的所有相似度：S = S\s(ri,r*)  移除候选区域rj对应的所有相似度：S = S\s(r*,rj)
    for k in key_to_delete:
        del S[k]

    # calculate similarity set with the new region  计算候选区域rt对应的相似度集合St,S = S∪St
    for k in filter(lambda a: a != (i, j), key_to_delete):
        n = k[1] if k[0] in (i, j) else k[0]
        S[(t, n)] = _calc_sim(R[t], R[n], imsize)

# 获取每一个候选区域的的信息  边框、以及候选区域size,标签
regions = []
for k, r in R.items():
    regions.append({
        'rect': (
            r['min_x'], r['min_y'],
            r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
        'size': r['size'],
        'labels': r['labels']
    })

# img：基于图的图像分割得到的候选区域   regions：Selective Search算法得到的候选区域
return img, regions
