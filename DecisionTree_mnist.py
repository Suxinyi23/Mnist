# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:10:33 2018

@author: suxin
"""



import numpy as np
import gzip
#from datetime import datetime

#导入数据
#C:\Users\suxin\.spyder\data\mnist
train_images_path='C:/Users/suxin/.spyder/data/mnist/train_images'
train_labels_path='C:/Users/suxin/.spyder/data/mnist/train_labels'
test_images_path='C:/Users/suxin/.spyder/data/mnist/test_images'
test_labels_path='C:/Users/suxin/.spyder/data/mnist/test_labels'


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量

def extract_images(input_file, is_value_binary):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))     
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
       # print magic, num_images, rows, cols
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows*cols)
        if is_value_binary:
            return (data!=0)
        else:
            return data
  

#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


x_train= extract_images(train_images_path,True)
y_train= extract_labels(train_labels_path)
x_test= extract_images(test_images_path,True)
y_test= extract_labels(test_labels_path)

y_train=y_train.reshape(len(y_train),1)

train=np.hstack((x_train,y_train))




def uniquecounts(rows):
    results = {}
    for row in rows:
        #计数结果在最后一列
        r = row[len(row)-1]
        if r not in results:results[r] = 0
        results[r]+=1
    return results # 返回一个字典

def entropy(rows):
    from math import log
    log2 = lambda x:log(x)/log(2)
    results = uniquecounts(rows)
    #开始计算熵的值
    ent = 0.0
    for r in results.keys():
        p = float(results[r])/len(rows)
        ent = ent - p*log2(p)
    return ent


#定义节点的属性
class decisionnode:
    def __init__(self,col = -1,value = None, results = None, tb = None,fb = None):
        self.col = col   # col是待检验的判断条件所对应的列索引值
        self.value = value # value对应于为了使结果为True，当前列必须匹配的值
        self.results = results #保存的是针对当前分支的结果，它是一个字典
        self.tb = tb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点



# 基尼不纯度


# 随机放置的数据项出现于错误分类中的概率
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp =0
    for k1 in counts:
        p1 = float(counts[k1])/total
        for k2 in counts: # 这个循环是否可以用（1-p1）替换？
            if k1 == k2: continue
            p2 = float(counts[k2])/total
            imp+=p1*p2
    return imp


# 改进giniimpurity


def giniimpurity_2(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts.keys():
        p1 = float(counts[k1])/total
        imp+= p1*(1-p1)
    return imp



#在某一列上对数据集进行拆分。可应用于数值型或因子型变量
def divideset(rows,column,value):
    #定义一个函数，判断当前数据行属于第一组还是第二组
    split_function = None
    if isinstance(value,int) or isinstance(value,float):
        split_function = lambda row:row[column] >= value
    else:
        split_function = lambda row:row[column]==value
    # 将数据集拆分成两个集合，并返回
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]

    return(set1,set2)


# 以递归方式构造树

def buildtree(rows,scoref = entropy):
    if len(rows)==0 : return decisionnode()
    current_score = scoref(rows)
    
    # 定义一些变量以记录最佳拆分条件
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    
    column_count = len(rows[0]) - 1
    for col in range(0,column_count):
        #在当前列中生成一个由不同值构成的序列
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1 # 初始化
        #根据这一列中的每个值，尝试对数据集进行拆分
        for value in column_values.keys():
            (set1,set2) = divideset(rows,col,value)
            
            # 信息增益
            p = float(len(set1))/len(rows)
            gain = current_score - p*scoref(set1) - (1-p)*scoref(set2)
            if gain>best_gain and len(set1)>0 and len(set2)>0:
                best_gain = gain
                best_criteria = (col,value)
                best_sets = (set1,set2)
                
    #创建子分支
    if best_gain>0:
        print 1
        trueBranch = buildtree(best_sets[0])  #递归调用
        print 1
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col = best_criteria[0],value = best_criteria[1],
                            tb = trueBranch,fb = falseBranch)
    else:
        return decisionnode(results = uniquecounts(rows))

# 决策树的显示
def printtree(tree,indent = ''):
    # 是否是叶节点
    if tree.results!=None:
        print str(tree.results)
    else:
        # 打印判断条件
        print str(tree.col)+":"+str(tree.value)+"? "
        #打印分支
        print indent+"T->",
        printtree(tree.tb,indent+" ")
        print indent+"F->",
        printtree(tree.fb,indent+" ")


# 对新的观测数据进行分类


def classify(observation,tree):
    if tree.results!= None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v,int) or isinstance(v,float):
            if v>= tree.value: branch = tree.tb
            else: branch = tree.fb
        else:
            if v==tree.value : branch = tree.tb
            else: branch = tree.fb
        return classify(observation,branch)
    
from datetime import datetime    
a = datetime.now()    
Decision_tree=buildtree(train,scoref=giniimpurity_2)#还有一个信息增益的方式可修改
b = datetime.now()
print "建树一共运行了%d秒"%((b-a).seconds)


#主体函数部分

a = datetime.now()
numTestSamples =x_test.shape[0]
matchCount = 0
test_num = numTestSamples
for i in xrange(test_num):

    predict = classify(x_test[i],Decision_tree)
    if max(predict,key=predict.get) == y_test[i]:
        matchCount += 1
    if i % 1000 == 0:
        print "完成%d张图片"%(i)
accuracy = float(matchCount) / test_num
b = datetime.now()
print "测试一共运行了%d秒"%((b-a).seconds)

print 'The classify accuracy is: %.2f%%' % (accuracy * 100)
