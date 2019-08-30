f=open('input.txt','r+')
f1=open('output.txt','w+')
wordcount={}
for i in f.read().split():
    if i not in wordcount:
        wordcount[i]=1
    else:
        wordcount[i]+=1
print(wordcount)
f1.write(str(wordcount))
