filename='f.txt'
f=open(filename,'r')
data=f.readlines()
sentences=[]
sentence=[]
start=-1
end=-1
entity=[]
filename2='pred_trans.txt'
f2=open(filename2,'w')
count=1
length=-1
sent=[]
for i,line in enumerate(data):
    if line.split():
        token,label=line.split()
        sent.append(token)
        ll=len(' '.join(sent))+1
        if data[i+1].split():
            _,label_n=data[i+1].split()
        print(token,label)
        if label[0]=='B':
           la,con=label.split('-')
           start=ll-len(token)
           end=ll
           entity.append(token)
        if label[0]=='I':
           end=ll
           entity.append(token)
        if label[0]=='B' and label_n[0]=='B':
           sentence.append(' '.join(['T'+str(count),con,str(start),str(end),' '.join(entity)]))
           start=-1
           end=-1
           entity=[]
        if label[0]=='I' and label_n[0]=='B':
           end=end+1
           entity.append(token)
           sentence.append(' '.join(['T'+str(count),con,str(start),str(end),' '.join(entity)]))
           start=-1
           end=-1
           entity=[]
        if label[0]=='O' and start>0:
           sentence.append(' '.join(['T'+str(count),con,str(start),str(end),' '.join(entity)]))
           start=-1
           end=-1
           entity=[]
    else:
        count=count+1
        sent=[]
        sentences.append('\n'.join(sentence))
        sentence=[]

f2.write('\n'.join(sentences))
    