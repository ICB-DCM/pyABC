from multiprocessing import Process, Queue, Pool




class A:
    def __call__(self, *args, **kwargs):
        return 1

    def __getstate__(self):
        raise Exception

a = A()



def feed(feed_q, n_jobs, n_proc):
    for _ in range(n_jobs):
        feed_q.put(1)

    for _ in range(n_proc):
        feed_q.put(None)


def work(feed_q, result_q, do_thing):
    while True:
        arg = feed_q.get()
        if arg == None:
            break
        res = do_thing(arg)
        result_q.put(res)


class ForkingMapper:
    def __init__(self, n_procs):
        self.n_procs = n_procs

    def map(self, f, lst):
        feed_q = Queue()
        result_q = Queue()

        feed_process = Process(target=feed, args=(feed_q, n_jobs, self.n_procs))

        worker_processes = [Process(target=work, args=(feed_q, result_q, f)) for _ in range(self.n_procs)]

        for proc in worker_processes:
            proc.start()

        feed_process.start()




def do_thing(arg):
    return 1





n_jobs = 20000



collected_results = []

for _ in range(n_jobs):
    collected_results.append(result_q.get())

feed_process.join()

for proc in worker_processes:
    proc.join()

print(collected_results)
assert len(collected_results) == n_jobs