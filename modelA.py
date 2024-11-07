import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def get_vocab(self, data, condition=None):
        freq = {}
        for i in data:
            if i[1] == condition:
                for j in i[0].split():
                    if j in freq:
                        freq[j]+=1
                    else:
                        freq[j]=1
        return freq

    def fit(self, dataset):
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test
        data = list(zip(self._train_X, self._train_y))
        self.ham = self.get_vocab(data=data,condition=1)
        self.spam = self.get_vocab(data=data,condition=0)
        self.vocab = set(self._train_X)
        
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        self.Nvoc = len(self.vocab)
        self.Nspam = len(set(self.spam))
        self.Nham = len(set(self.ham))
        pass
    
    def inference(self, message):
        p_spam = self._train_y.count(0)/len(self._train_y)
        p_ham = 1 - p_spam
        new = []
        for i in message.split():
            new.append(re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',"",i))
        for i in range(len(new)):
            new[i]=new[i].lower()
        
        
        ps=1
        
        for i in new:
            if i not in self.spam:
                ps*=self.alpha/(self.Nham+self.alpha*self.Nvoc)
            else:
                ps*=(self.alpha+self.spam[i])/(self.spam[i]+self.alpha*self.Nvoc)
        
        ph=1
        for i in new:
            if i not in self.ham:
                ph*=self.alpha/(self.Nham+self.alpha*self.Nvoc)
            else:
                ph*=(self.alpha+self.ham[i])/(self.ham[i]+self.alpha*self.Nvoc)
                
        pspam = p_spam*ps
        pham = p_ham*ph
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        match = 0
        wrong = 0

        for message, truth in zip(self._val_X, self._val_y):
            #print(message, truth)
            res = self.inference(message)
            if res == self.num2label[truth]:
                match+=1
            else:
                wrong+=1
       
        val_acc = round(100*match/(match+wrong), 2)
        return val_acc

    def test(self):
        match = 0
        wrong = 0
        for message, truth in zip(self._test_X, self._test_y):
            res = self.inference(message)
            if res == self.num2label[truth]:
                match+=1
            else:
                wrong+=1
  
        test_acc = round(100*match/(match+wrong), 2)
        return test_acc


