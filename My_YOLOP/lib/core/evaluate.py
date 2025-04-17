import numpy as np


class SegmentationMetric(object):
    '''
    imgLabel [batch_size, height(144), width(256)]
    confusionMatrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    '''  
    def __init__(self, numClass=2):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2) #numClass x numClass 的零矩阵
        
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()       
        return acc
    
    def lineAccuracy(self):
        # Acc[0] = TN / TN + FP
        # Acc[1] = TP / FN + TP
        Acc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-22) #axis=1 表示按行求和
        return Acc[1]
    
    def classPixelAccuracy(self):
        # acc = TP / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc
    
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = [TN, TP]; Union = [(FP + FN + TN), (TP + FP + FN)]
        # IoU = [TN, TP] / [(FP + FN + TN), (TP + FP + FN)]
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0 #考虑到0/0的情况
        mIoU = np.nanmean(IoU)
        return mIoU
    
    def IntersectionOverUnion(self):
        # IoU[1] = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # 生成有效像素掩码mask
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # 生成唯一标签数组索引label = 真实标签 * numClass + 预测标签
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        # 计算每对标签出现的次数
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIoU = [(TP +　FN) / (TP + FP + TN + FN)] * [TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        iu = intersection / union
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
        
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))





# Plots ----------------------------------------------------------------------------------------------------------------
