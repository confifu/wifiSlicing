from multiprocessing import Pool, Process, Queue
import os

id = os.getpid()
def f(x, y, q):
    print('parent process: ', os.getppid())
    print('process id: ', os.getpid())
    print(x, y)
    q.put(x*x)

def run():
    q = Queue()



    ps = []
    for i in range(5):
        p = Process(target=f, args=(0, 1, q))
        p.start()
        ps.append(p)

    for i, p in enumerate(ps):
        p.join()
        print("Process ", i, " joined")
        print(q.get())

            



    '''
    with Pool(5) as p:
        print(p.starmap(f, [(1, 1), (2, 2),  (3, 3)]))
    '''