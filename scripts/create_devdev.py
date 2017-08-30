import csv

with open('F:/traindev/devset.csv',mode='rt', encoding='utf-8') as csvfile:
    ofile = open('F:/traindev/devset_reduced.csv', mode='wt', encoding='utf-8')

    reader = csv.DictReader(csvfile)
    header = reader.fieldnames
    writer = csv.DictWriter(ofile, header, lineterminator='\n')
    writer.writeheader()

    current_mr = ''
    for row in reader:
        mr = row['mr']
        ref = row['ref']

        if not current_mr == mr:
            oline = {'mr': mr, 'ref': ref}
            writer.writerow(oline)
            current_mr = mr

    ofile.close()