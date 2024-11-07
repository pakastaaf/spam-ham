import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        self.x = []
        for i in self._x:
            new = re.findall(r"\w+", i.lower()) 
            new = " ".join(j for j in new)
            self.x.append(new)
        for i in self._y:
            if i == "spam":
                self.label2num[i]=0
            else:
                self.label2num[i]=1
        self.y = []
        for i in self._y:
            self.y.append(self.label2num[i])
        
        for i in self.y:
            if i == 0:
                self.num2label[i]="spam"
            else:
                self.num2label[i]="ham"
        pass

    def split_dataset(self, val=0.1, test=0.1):
        indices = np.arange(0, len(self._x))
        np.random.seed(1)
        np.random.shuffle(indices)
        
        val_i = indices[:round(val*len(self._x))]
        test_i = indices[round(val*len(self._x)):round((val+test)*len(self._x))] 
        train_i = indices[round((test+val)*len(self._x)):]
        
        train_X = []
        for i in train_i:
            train_X.append(self.x[i])
        train_y = []
        
        for i in train_i:
            train_y.append(self.y[i])
            
        test_X = []
        for i in test_i:
            test_X.append(self.x[i])
        test_y = []
        
        for i in test_i:
            test_y.append(self.y[i])
        
        val_X = []
        for i in val_i:
            val_X.append(self.x[i])
        val_y = []
        
        for i in val_i:
            val_y.append(self.y[i])

        self.train=[train_X, train_y]
        self.val=[val_X,val_y]
        self.test=[test_X, test_y]
        pass
