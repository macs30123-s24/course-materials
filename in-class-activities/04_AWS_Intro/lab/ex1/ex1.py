import random

# Some computationally expensive code
n = 10
res = []
for i in range(n):
    res.append( random.random() )

# Save results
file = 'results.txt'
with open(file, 'w') as f:
    for x in res:
        f.write(str(x) + '\n')