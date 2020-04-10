import os

with open('neg.txt', 'w') as n:
    for f in os.listdir('neg'):
        n.write(f'neg/{f}\n')

