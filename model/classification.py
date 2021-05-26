# 分类器(分类vec，结构规整->独热编码)

# TODO:

# 核kmeans聚类模型

# GMM-EM

# 核化软间隔svm
# 对抗投票输出



class classification:
    #
    def __init__(self, classifier = '', **kwargs) -> None:
        self.classifier = classifier
        for item in kwargs.items():
            print(item)
            
a = classification('asd', time = 6, name = 6)
# print(a.time)
# 
# print(a.name)