cimport cython
import numpy as np
cimport numpy as np


def prime_counter_cy(range_from: int, range_til: int):
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

@cython.cdivision(True)
cpdef prime_counter_cy_opt(int range_from, int range_til):
  """ Returns the number of found prime numbers using range"""
  cdef int prime_count = 0
  cdef int num
  cdef int divnum
  range_from = range_from if range_from >= 2 else 2
  for num in range(range_from, range_til + 1):
    for divnum in range(2, num):
      if ((num % divnum) == 0):
        break
    else:
      prime_count += 1
  return prime_count


cpdef add_arrs_cy(np.ndarray[double] a, np.ndarray[double] b):
  #cdef int i
  #cdef int n = a.shape[0]
  #cdef np.ndarray[double] c = np.zeros(n)
  #for i in range(n):
  #  c[i] = a[i] + b[i]
  cdef np.ndarray[double] c = a + b
  return c