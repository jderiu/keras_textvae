import csv


dev_fname = 'F:/traindev/devset.csv'
reader = csv.DictReader(open(dev_fname, encoding='utf-8', mode='rt'))

output_fname = 'F:/traindev/test_output_5.txt'
oput = open(output_fname,mode='rt', encoding='utf-8')

inputs_raw = []
for row in reader:
    i1 = row['mr']
    inputs_raw.append(i1)

outputs_raw = []
for line in oput:
    outputs_raw.append(line.replace('\n', ''))

ofile = open('outputs.csv', 'wt', encoding='utf-8', newline='\n')
writer = csv.DictWriter(ofile, fieldnames=['mr', 'ref'], delimiter=',')
writer.writeheader()
for i, o in zip(inputs_raw, outputs_raw):
    writer.writerow({'mr': i, 'ref': o})

ofile.close()

