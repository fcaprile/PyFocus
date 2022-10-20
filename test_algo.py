def f():
    return 1

class A:
    fu = f()
    
    def exe(self):
        return self.fu    

print(A().exe())