from threading import Thread


def a():
    global name 
    name = "hoang"
def b():
    print (name)

th1 = Thread(target=a)
th1.start()
th2 = Thread(target=b)
th2.start()