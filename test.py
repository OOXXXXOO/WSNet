class A():

    def __init__(self):
        self.a=1
        print('A:',self.a)

class B(A):
    def __init__(self):
        A.__init__(self)
        self.b=2
        print('B:',self.b)

class C(A):
    def __init__(self):
        self.c=3
        print('C:',self.c)

class D(B,C):
    def __init__(self,a):
        self.d=a
        B.__init__(self)
        C.__init__(self)
        print('D:',self.d)

def main():
    print(D.mro())
    d=D(1)
    print(d.a)


if __name__ == '__main__':
    main()
    