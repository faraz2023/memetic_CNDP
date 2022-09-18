#python setup.py build_ext --inplace


import time
from primes import prime_counter_cy, prime_counter_cy_opt, add_arrs_cy
import numpy as np
def prime_count_vanilla_range(range_from: int, range_til: int):
  """ Returns the number of found prime numbers using range"""
  prime_count = 0
  range_from = range_from if range_from >= 2 else 2
  for num in range(range_from, range_til + 1):
    for divnum in range(2, num):
      if ((num % divnum) == 0):
        break
    else:
      prime_count += 1
  return prime_count

def add_arrs(a, b):
  n = a.shape[0]
  c = np.zeros(n)
  for i in range(n):
    c[i] = a[i] + b[i]
  return c


if __name__ == "__main__":
    #time function
    start = time.time()
    prime_count = prime_count_vanilla_range(1, 10000)
    end = time.time()
    print("py Time: {} Prime count: {prime_count}".format(end-start, prime_count=prime_count))

    #time function
    start = time.time()
    prime_count = prime_counter_cy(1, 10000)
    end = time.time()
    print("cy Time: {} Prime count: {prime_count}".format(end-start, prime_count=prime_count))

    #time function
    start = time.time()
    prime_count = prime_counter_cy_opt(1, 10000)
    end = time.time()
    print("cy_opt Time: {} Prime count: {prime_count}".format(end-start, prime_count=prime_count))


    k = int(1e7)
    a = np.random.rand(k)
    b = np.random.rand(k)


    #time function
    start = time.time()
    c = add_arrs_cy(a, b)
    end = time.time()
    print("ADD cy Time: {} ".format(end-start))

    #time function
    start = time.time()
    c = a  + b
    end = time.time()
    print("ADD py Time: {} ".format(end-start))

