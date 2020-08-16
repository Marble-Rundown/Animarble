import csv, os

def create_file(file_name, file_type, n=0):
    destination = './outputs/{0}{1}.{2}'.format(file_name, f' ({n})', file_type)       # Add:    if n != 0 else ''    after f' ({n})' if you don't want the first file to have a number
    if not os.path.isfile(destination):
        return open(destination, 'w+')
    else:
        return create_file(file_name, file_type, n+1)

destination = create_file('Bob_Cropped_converted', 'csv')
with open('outputs/BobCropped (0).csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in csv_reader:
            if 'timestamp' not in row:
                #print([float(x) for x in row])
                line = [float(x) for x in row]
                destination.write(f'{line[0]},{line[1] - 90},{line[2] - 90},90,90\n')
            else:
                destination.write('timestamp,tilt,pan,tilt_setpoint,pan_setpoint\n')
