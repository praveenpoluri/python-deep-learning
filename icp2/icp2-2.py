print('enter the string:')
str = input()
n=len(str)

def string_alternative():
    l=[]
    for i in range(0,n):
        if(i%2!=0):
            x=str[i]
            l.append(x)

    y=''.join(l)
    print(y)
















string_alternative()
