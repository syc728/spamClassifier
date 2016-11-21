import docclass
import os
import glob

def train_naivebayes():
    train_path = os.getcwd() + '\\train\\*.txt'
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
    
    test_path = os.getcwd() + '\\test\\*.txt'
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
    
    P = 1.0 * TP / (TP + FP)
    R = 1.0 * TP / (TP + FN)
    
    print "Precision : ", P
    print "Recall : ", R  

def train_fisher():
    train_path = os.getcwd() + '\\train\\*.txt'
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
    
    test_path = os.getcwd() + '\\test\\*.txt'
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
    
    P = 1.0 * TP / (TP + FP)
    R = 1.0 * TP / (TP + FN)
    
    print "Precision : ", P
    print "Recall : ", R  

if __name__ == '__main__':
    print "Naive Bayes:"
    #train_naivebayes()
    print "Fisher:"
    train_fisher()