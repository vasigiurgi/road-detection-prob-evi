#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
#import pdb


# Read paths of validation split
def read_path(dir_name, file_name):
    dir_file_name=os.path.join(dir_name,file_name)
    f = open(dir_file_name, 'r')
    list_path_names = [line.replace("\n", "") for line in f.readlines()]
    f.close()
    return list_path_names

#zero unpadding
def zero_unpadding(array_img, h_actual, w_actual, img_size):  
    dh=int((img_size[0]-h_actual)/2)
    dw=int((img_size[1]-w_actual)/2)
    zero_unpadded_img=array_img[dh:dh+h_actual,dw:dw+w_actual]
    return zero_unpadded_img

# change from cluster path to local path
def clu_to_local(file_path):
    if file_path.startswith ("/home2020"):
        file_path=file_path.split("mngeletu/")
        file_path=os.path.join("/home/deeplearning-miam/Documents/", file_path[1])
    return file_path

def getGroundTruth(fileNameGT):
    '''
    Returns the ground truth maps for roadArea and the validArea 
    :param fileNameGT:
    '''
    # Read GT
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    #full_gt = cv2.imread(fileNameGT, cv2.CV_LOAD_IMAGE_UNCHANGED)
    full_gt = cv2.imread(fileNameGT, -1)
    #attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea =  full_gt[:,:,0] > 0
    validArea = full_gt[:,:,2] > 0

    return roadArea, validArea

def evalExp(gtBin, cur_prob, thres, validMap = None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_prob:
    :param thres:
    :param validMap:
    '''

    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
    
    #Merge validMap with validArea
    if validMap!=None: # .all() added
        if validArea!=None:
            validMap = (validMap == True) & (validArea == True)
    elif validArea.all()!=None:
        validMap=validArea

    # histogram of false negatives
    if validMap.all()!=None:
        fnArray = cur_prob[(gtBin == True) & (validMap == True)]
    else:
        fnArray = cur_prob[(gtBin == True)]
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)];
    
    if validMap.all()!=None:
        fpArray = cur_prob[(gtBin == False) & (validMap == True)]
    else:
        fpArray = cur_prob[(gtBin == False)]
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    #posNum = fnArray.shape[0]
    #negNum = fpArray.shape[0]
    if validMap.all()!=None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum


def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
    '''

    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''

    #Calc missing stuff
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP


    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    precision =  totalTP / (totalTP + totalFP + 1e-10)
    
    selector_invalid = (recall==0) & (precision==0)
    recall = recall[~selector_invalid]
    precision = precision[~selector_invalid]
        
    #Pascal VOC average precision
    AvgPrec = 0
    counter = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        if ind == None:
            continue
        pmax = max(precision[ind])
        AvgPrec += pmax
        counter += 1
    AvgPrec = AvgPrec/counter
    
    
    # F-measure operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    MaxF= F[index]
    
    #recall_bst = recall[index]
    #precision_bst =  precision[index]

    TP = totalTP[~selector_invalid][index]   # New: selector_invalid applied
    TN = totalTN[~selector_invalid][index]   #    >>
    FP = totalFP[~selector_invalid][index]   #    >>
    FN = totalFN[~selector_invalid][index]   #    >>
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    #ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
    prob_eval_scores  = calcEvalMeasures(valuesMaxF)
    prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF

    #prob_eval_scores['totalFN'] = totalFN
    #prob_eval_scores['totalFP'] = totalFP
    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    #prob_eval_scores['precision_bst'] = precision_bst
    #prob_eval_scores['recall_bst'] = recall_bst
    thresh = thresh[~selector_invalid] # New, invalid selector
    prob_eval_scores['thresh'] = thresh
    if thresh.all() != None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh

    #return a dict
    return prob_eval_scores


def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    #TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    
    outDict =dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['ER'+ tag] = (FP+FN)/(P+N)
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict

class dataStructure: 
    '''
    All the defines go in here!
    '''
    
    cats = ['um_road', 'umm_road', 'uu_road']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_propertyList = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp','totalPosNum','totalNegNum', 'FPR_wp', 'FNR_wp','BestThresh', 'TP_wp', 'FP_wp', 'FN_wp', 'TN_wp', 'ER_wp' ] 

#########################################################################
# function that does the evaluation
def mainEval(result_dir, gt_fileList_all, debug = False):

    '''
    main method of evaluateRoad
    :param result_dir: directory with the result propability maps, e.g., /home/elvis/kitti_road/my_results
    :param debug: debug flag (OPTIONAL)
    '''
    
    print("Starting evaluation ..." ) 
    print ("Available categories are:", dataStructure.cats)
    
    f= open('stage1_evaluation_metrics.txt', 'w')
    thresh = np.array(range(0,256))/255.0
    
    assert os.path.isdir(result_dir), 'Cannot find result_dir: %s ' %result_dir
    
    # In the submission_dir we expect the probmaps! 
    submission_dir = result_dir
    assert os.path.isdir(submission_dir), 'Cannot find %s, ' %submission_dir
    
    # init result
    prob_eval_scores = [] # the eval results in a dict
    eval_cats = [] # saves all categories at were evaluated

    # New, evaluating for urban road 
    ur_totalFP=np.zeros( thresh.shape )
    ur_totalFN=np.zeros( thresh.shape )
    ur_totalPosNum=0
    ur_totalNegNum=0
    #New, evaluating for urban road category
    for cat in dataStructure.cats:
        print ("Execute evaluation for category ...", cat)
        #f.write('%s %s:\n' %("Evaluation metrics for category", cat))
        gt_fileList=[]
        for fname in gt_fileList_all:
            fname_key=fname.split('/')[-1]
            if fname_key.startswith(cat) and fname_key.endswith(dataStructure.gt_end):
                gt_fileList.append(fname)

        assert len(gt_fileList)>0, 'Error reading ground truth'
        # Init data for categgory
        category_ok = True # Flag for each cat
        totalFP = np.zeros( thresh.shape )
        totalFN = np.zeros( thresh.shape )
        totalPosNum = 0
        totalNegNum = 0
        
        for fn_curGt in gt_fileList:
            
            file_key = fn_curGt.split('/')[-1].split('.')[0]
            if debug:
                print ("Processing file: ", file_key)
            
            # Read GT
            cur_gt, validArea = getGroundTruth(fn_curGt)
                        
            # Read probmap and normalize
            fn_curProb = os.path.join(submission_dir, file_key + dataStructure.prob_end)
            
            if not os.path.isfile(fn_curProb):
                print ("Cannot find file: %s for category %s." %(file_key, cat))
                print ("--> Will now abort evaluation for this particular category.")
                category_ok = False
                break
            
            cur_prob = cv2.imread(fn_curProb,0)
            cur_prob = np.clip( (cur_prob.astype('f4'))/(np.iinfo(cur_prob.dtype).max),0.,1.)  # check the max of dtype
            
            FN, FP, posNum, negNum = evalExp(cur_gt, cur_prob, thresh, validMap = None, validArea=validArea)
            #pdb.set_trace()
            assert FN.max()<=posNum, 'BUG @ poitive samples'
            assert FP.max()<=negNum, 'BUG @ negative samples'
            
            # collect results for whole category
            totalFP += FP
            totalFN += FN
            totalPosNum += posNum
            totalNegNum += negNum
        
        if category_ok:
            print ("Computing evaluation scores...")
            # Compute eval scores!
            prob_eval_scores.append(pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = thresh))
            eval_cats.append(cat)
            
            factor = 100
            for property in dataStructure.eval_propertyList:
                if property not in ['BestThresh','TP_wp','totalPosNum','totalNegNum', 'FP_wp', 'FN_wp', 'TN_wp']:
                    pass
                    #f.write('%s: %4.2f\n' %(property, prob_eval_scores[-1][property]*factor))
                elif property == 'BestThresh':
                    pass
                    #f.write('%s: %s\n' %(property, prob_eval_scores[-1][property]))
            print ("Finished evaluating category: %s " %(eval_cats[-1],)) 
        
        #New, evaluation for urban road category
        ur_totalFP+=totalFP
        ur_totalFN+=totalFN
        ur_totalPosNum+=totalPosNum
        ur_totalNegNum+=totalNegNum
    print ("Execute evaluation for category urban road (comulative) ..." )
    f.write('%s\n' %("Evaluation metrics for category urban road (comulative):"))
    ur_prob_eval_scores=pxEval_maximizeFMeasure(ur_totalPosNum, ur_totalNegNum, ur_totalFN, ur_totalFP, thresh = thresh)
    factor = 100
    for property in dataStructure.eval_propertyList:
        if property not in ['BestThresh','totalPosNum','totalNegNum','TP_wp', 'FP_wp', 'FN_wp', 'TN_wp']:
            f.write('%s: %4.2f\n' %(property, ur_prob_eval_scores[property]*factor))  
        else:
            f.write('%s: %s\n' %(property, ur_prob_eval_scores[property]))
    f.close()
    print ("Finished evaluation category: urban road")

    # New, evaluating for urban road category
    
    if len(eval_cats)>0:     
        print ("Successfully finished evaluation for %d categories: %s " %(len(eval_cats),eval_cats))
        return True
    else:
        print ("No categories have been evaluated!")
   
    return False
    
