#print('enter a list:')
#a=input()

a=[( 'John', ('Physics', 80)) , ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)), ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]

d= {}

for k, v in a:
    if k not in d:
        d[k] = v
    else:
        d[k] = [d[k]]
        d[k].append(v)
print(d)