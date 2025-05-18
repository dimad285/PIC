import os

folder = 'src'

total = 0

list = os.listdir(folder)
for i in list:
    i = folder+'/'+i
    if i.endswith('.py'):
        print(i)
        with open(i, 'r') as f:
            lines = len(f.readlines())
            print(lines)
            total += lines

print(f'Number of lines in {folder}:', total)