import classifyText

class Message:
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
    
    def __repr__(self):
        return self.msg

    def classifyMessage(self,label_classifier,targets_classifiers, name_classifers):
        flag, response = classifyText.classify_text(self.msg, label_classifier, targets_classifiers, name_classifers, treshold_bot = 0.4,verbose=True )
        return flag, response

class User:
    def __init__(self, id, username = None):
        self.id = id
        if(username == None):
            self.username = "Unknown"
        else:
            self.username = username
        self.n_badWords = 0
        self.n_goodWords = 0

    def __str__(self):
        return self.username

    def addBadWord(self):
        self.n_badWords += 1
        self.n_goodWords = 0
    
    def addGoodWord(self):
        self.n_goodWords += 1