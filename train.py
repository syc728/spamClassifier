import docclass
import os
import glob

def score(TP,FN,FP,TN):
    P = 1.0 * TP / (TP + FP)
    R = 1.0 * TP / (TP + FN)
    
    print "Precision : ", P
    print "Recall : ", R  

def train_naivebayes(train_path, test_path):
    cl = docclass.naivebayes(docclass.getwords)
    
    for filename in glob.glob(train_path):
        with open(filename,'r') as f:
            f = f.read()
            label = filename.split('.')[3]
            cl.train(f,label)
            
    print "Train Done!"
    
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    
    cl.setthreshold('ham',2.4)
    for filename in glob.glob(test_path):
        with open(filename,'r') as f:
            f = f.read()
            label = filename.split('.')[3]
            predict = cl.classify(f,default = 'Unknown')
            if label == 'spam' and predict == 'spam':
                TP += 1
            elif label == 'spam' and predict == 'ham':
                FN += 1
            elif label == 'ham' and predict == 'spam':
                FP += 1
            elif label == 'ham' and predict == 'ham':
                TN += 1
            else:
                print predict, label
    
    print "Test Done!"
    
    score(TP,FN,FP,TN)
    

def train_fisher(train_path, test_path):
    cl = docclass.fisherclassifier(docclass.getwords)
    
    for filename in glob.glob(train_path):
        with open(filename,'r') as f:
            f = f.read()
            label = filename.split('.')[3]
            cl.train(f,label)
            
    print "Train Done!"
    
    TP = 0.0
    FN = 0.0
    FP = 0.0
    TN = 0.0
    
    for filename in glob.glob(test_path):
        with open(filename,'r') as f:
            f = f.read()
            label = filename.split('.')[3]
            predict = cl.classify(f)
            if label == 'spam' and predict == 'spam':
                TP += 1
            elif label == 'spam' and predict == 'ham':
                FN += 1
            elif label == 'ham' and predict == 'spam':
                FP += 1
            elif label == 'ham' and predict == 'ham':
                TN += 1
            else:
                print predict, label
    
    print "Test Done!"
    
    score(TP,FN,FP,TN)

if __name__ == '__main__':
    train_path = os.getcwd() + '\\train\\*.txt'
    test_path = os.getcwd() + '\\test\\*.txt'
    print "Naive Bayes:"
    train_naivebayes(train_path, test_path)
    print "Fisher:"
    train_fisher(train_path, test_path)