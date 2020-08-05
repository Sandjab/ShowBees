import multiprocessing as mp
import librosa
from os import fspath
from os import getpid
import timeit
from functools import partial
from itertools import repeat

def worker(path, iterations):
    total = 0;
    for i in range(iterations):
        sig, sr = librosa.core.load(fspath(path))
        total += len(sig) % 4
    return total

def run_pool(processes, files, iterations):
    with mp.Pool(processes) as pool:
        print("Start Pool")
        results = pool.map(partial(worker, iterations=iterations), files)
        #results = pool.starmap(worker, (files, iterations))
        print("End Pool")
        print(results)
        print('------------------')
        for r in results:
            print(r)
        
def run_pool_async(processes, files, iterations):
    with mp.Pool(processes) as pool:
        results = [pool.apply_async(worker, (f, iterations)) for f in files]  
        print([res.get(timeout=30) for res in results])

def sequential(files, iterations): 
    print(iterations)
    total = 0;
    for f in files:
        for i in range(iterations):
            sig, sr = librosa.core.load(fspath(f))
            total += len(sig) % 4
    return total

        

def run(wavs, iterations=10, nprocs=1, repeat=1):
    print("Sequential:", timeit.timeit(lambda: sequential(wavs, iterations), number=repeat))
    print("Parallel", nprocs, ":", timeit.timeit(lambda: run_pool(nprocs, wavs, iterations), number=repeat))

    #print("Async", nprocs, ":", timeit.timeit(lambda: run_pool_async(nprocs, wavs), number=repeat))
   