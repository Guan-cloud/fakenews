filename='final_guan.txt'
f1=open(filename,'rb')
data=f1.read()
print(data)

Expect23=['(', 'IQR', ',', '38', '-', '69', ';', 'range', '20', '-', '95', 'years', ')', ',', 'of', 'whic  h', '67', '', '(', '%', ')', 'were', 'male']
But22   =['O', 'B-CON', 'O', 'B-CON', 'O', 'O', 'O', 'B-ACT', 'B-CON', 'O', 'O', 'B-CON', 'O', 'O', 'O', 'O', 'B-CON', 'O', 'O', 'O', 'O', 'B-CON']

# filename='fakenews.txt'
# f=open(filename,'rb')
# data=f.read()
# print(data)
# data=data.replace(b'\r\n',b'\n')
# filename2='final2.txt'
# f2=open(filename2,'wb')
# f2.write(data)
