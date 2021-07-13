filename='final.txt'
f=open(filename,'r')
filename2=('fake6.txt')
f2=open(filename2,'w')
data=f.readlines()
print("data:")
print(data)
acc=[]
for da in data:
    qwe=[]
    print("da:",da)
    line=da.split()
    for li in line:
        print("li:")
        print(li)
        li='"'+li+'"'
        qwe.append(li)
        print("qwe:")
        print(qwe)
    acc.append(' '.join(qwe))
qsc='\n'.join(acc)
f2.write(qsc)