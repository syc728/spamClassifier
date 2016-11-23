from math import tanh
import sqlite3 as sqlite
import re

def dtanh(y):
    return 1.0-y*y
    
def getwords(doc):
    splitter = re.compile('\\W*')
    
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]
    
    return words

class searchnet:
    def __init__(self, dbname):
        self.con = sqlite.connect(dbname)
    
    def __del__(self):
        self.con.close()
      
    def maketables(self):
        self.con.execute('create table hiddennode(create_key)')
        self.con.execute('create table wordhidden(fromid, toid, strength)')
        self.con.execute('create table hiddenans(fromid, toid, strength)')
        self.con.commit()
        
    def getstrength(self, fromid, toid, layer):
        if layer == 0:
            table = 'wordhidden'
        else:
            table = 'hiddenans'
        
        res = self.con.execute('select rowid from %s where fromid=%d and toid=%d' % (table, fromid, toid)).fetchone()
        
        if res == None:
            if layer == 0:
                return -0.2
            if layer == 1:
                return 0
        return res[0]
            
    
    def setstrength(self, fromid, toid, layer, strength):
        if layer == 0:
            table = 'wordhidden'
        else:
            table = 'hiddenans'
        
        res = self.con.execute('select rowid from %s where fromid=%d and toid=%d' % (table, fromid, toid)).fetchone()
        
        if res == None:
            self.con.execute('insert into %s (fromid, toid, strength) values (%d,%d,%f)' % (table, fromid, toid, strength))
        else:
            rowid = res[0]
            self.con.execute('update %s set strength=%f where rowid=%d' % (table, strength, rowid))
    
    def generatehiddennode(self, wordids, ansids):
        if len(wordids) > 3:
            return None
        createkey = '_'.join(sorted([str(wi) for wi in wordids]))
        res = self.con.execute("select rowid from hiddennode where create_key='%s'" % createkey).fetchone()
        
        if res == None:
            cur = self.con.execute("insert into hiddennode (create_key) values ('%s')" % createkey)
            hiddenid = cur.lastrowid
            
            for wordid in wordids:
                self.setstrength(wordid, hiddenid, 0, 1.0/len(wordids))
            for ansid in ansids:
                self.setstrength(hiddenid, ansid, 1,0.1)
            self.con.commit()
    
    def getallhiddenids(self, wordids, ansids):
        l1 = {}
        for wordid in wordids:
            cur = self.con.execute('select toid from wordhidden where fromid=%d' % wordid)
            for row in cur:
                l1[row[0]] = 1
        for ansid in ansids:
            cur = self.con.execute('select fromid from hiddenans where toid=%d' % ansid)
            for row in cur:
                l1[row[0]] = 1
        return l1.keys()
        
    def setupnetwork(self, wordids, ansids):
        self.wordids = wordids
        self.hiddenids = self.getallhiddenids(wordids, ansids)
        self.ansids = ansids
        
        self.ai = [1.0] * len(self.wordids)
        self.ah = [1.0] * len(self.hiddenids)
        self.ao = [1.0] * len(self.ansids)
        
        self.wi = [[self.getstrength(wordid, hiddenid, 0) for hiddenid in self.hiddenids] for wordid in self.wordids]
        self.wo = [[self.getstrength(hiddenid, ansid, 1) for ansid in self.ansids] for hiddenid in self.hiddenids]
        
    def feedforward(self):
        for i in range(len(self.wordids)):
            self.ai[i] = 1.0
        
        for j in range(len(self.hiddenids)):
            sum = 0.0
            for i in range(len(self.wordids)):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)
            
        for k in range(len(self.ansids)):
            sum = 0.0
            for j in range(len(self.hiddenids)):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)
            
        return self.ao[:]
    
    def getresult(self, wordids, ansids):
        self.setupnetwork(wordids, ansids)
        return self.feedforward()

    def backPropagate(self, targets, N=0.5):
        output_deltas = [0.0] * len(self.ansids)
        for k in range(len(self.ansids)):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dtanh(self.ao[k]) * error
            
        hidden_deltas = [0.0] * len(self.hiddenids)
        for j in range(len(self.hiddenids)):
            error = 0.0
            for k in range(len(self.ansids)):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error
        
        for j in range(len(self.hiddenids)):
            for k in range(len(self.ansids)):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change
        
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change
    
    def updatedatabase(self):
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                self.setstrength(self.wordids[i], self.hiddenids[j], 0, self.wi[i][j])
        
        for j in range(len(self.hiddenids)):
            for k in range(len(self.ansids)):
                self.setstrength(self.hiddenids[j], self.ansids[k], 1, self.wo[j][k])
        self.con.commit()
    
    def trainquery(self, wordids, ansids, selectedans):
        self.generatehiddennode(wordids, ansids)
        
        self.setupnetwork(wordids, ansids)
        self.feedforward()
        targets = [0.0] * len(ansids)
        targets[ansids.index(selectedans)] = 1.0
        self.backPropagate(targets)
        self.updatedatabase()
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            