class A():
    def __init__(self,A=0):
        if A==0:
            def a(self):
                print('a')
        if A==1:
            def a(self):
                print('b')

a=A(1)
a.a()
aa=A(0)
a.a()
    