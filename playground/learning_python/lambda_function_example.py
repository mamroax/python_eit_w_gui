def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(3)
print(mydoubler(21))
