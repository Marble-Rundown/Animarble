import csv, os, argparse


#############################
#         Constants         #
#############################
def check_str(value):
    if type(value) != str:
        raise argparse.ArgumentTypeError(f"'{value}' is not a string")
    return value
def check_float(value):
    if type(float(value)) != float:
        raise argparse.ArgumentTypeError(f"'{value}' is not a float")
    return value
ap = argparse.ArgumentParser()
# ap.add_argument('-f', '--function', required=True, type=check_str, help='The name of the overall void function')
ap.add_argument('-s', '--scale', required=True, help='A multiplier for video speed')
ap.add_argument('-l', '--leftFile', required=True, help='Left file')
ap.add_argument('-r', '--rightFile', required=True, help='Right file')

args = vars(ap.parse_args())


leftFile = args['leftFile']
rightFile = args['rightFile']

functionName = os.path.splitext(os.path.basename(leftFile))[0].split("_")[0]
print(functionName)
print(leftFile)


#OFFSET = 0      # Left Video:   -----------------------...
                # Right Video:               ----------...
                #               |<- offset ->|
                # if Left Video starts after Right Video, offset is negative
K = 2       # number of 33ms time intervals per interval in the final file
OVERALL_FUNCTION = functionName #args['function']
MARBLE_FUNCTION = 'moveMarbles'       # Name of the Arduino function that rotates the marbles

MIN_TILT = 60
MAX_TILT = 110
MIN_PAN = 0
MAX_PAN = 150

MULTIPLIER = float(args['scale'])


#############################
#            Main           #
#############################
def main():
    #print('yay')
    # bobFile, right = most_recent_file('left'), most_recent_file('right')
    left, right = csv_to_list(leftFile), csv_to_list(rightFile)

    left_interval, right_interval = sum([left[i+1][0] - left[i][0] for i in range(len(left) - 1)]) / len(left), sum([right[i+1][0] - right[i][0] for i in range(len(right) - 1)]) / len(right)
    interval = (left_interval + right_interval) / 2

    output = create_file('final', 'h')
    output.write(f'void {OVERALL_FUNCTION}()')
    output.write('{\n')

    last_left, last_right = 0, 0
    left_ended, right_ended = False, False
    line = 0
    timestamp = 0
    while len(left) > 0 or len(right) > 0:
        if len(left) < K:       # check if list will soon be emptied
            left_ended = True
            diff = K - len(left)
            for i in range(diff):
                left.append([timestamp, last_left[0], last_left[1], 0])
        if len(right) < K:       # check if list will soon be emptied
            right_ended = True
            diff = K - len(right)
            for i in range(diff):
                right.append([timestamp, last_right[0], last_right[1], 0])
                
        left_tilt = int(round(sum([left[i][1] for i in range(K)]) / K))
        left_pan = int(round(sum([left[i][2] + left[i][3] for i in range(K)]) / K))
        right_tilt = int(round(sum([right[i][1] for i in range(K)]) / K))
        right_pan = int(round(sum([right[i][2] + right[i][3] for i in range(K)]) / K))
        
        last_left, last_right = (left_tilt, left_pan), (right_tilt, right_pan)
        timestamp = line * interval * K * MULTIPLIER

        output.write('  // Row {0}{1}{2}\n'.format(line * K + 2, ', left ended' if left_ended else '', ', right ended' if right_ended else ''))
        output.write(f'  {MARBLE_FUNCTION}({int(round(timestamp))}, {left_pan}, {left_tilt}, {right_pan}, {right_tilt});\n\n')

        left, right = left[K:], right[K:]       # Pop first K elements
        line += 1
    output.write('}')


#############################
#         Functions         #
#############################
def most_recent_file(keyword):
    files = []
    for fname in os.listdir('outputs/'):
        if keyword in fname:
            files.append(fname)
    return max(files)

def csv_to_list(file_name):
    contents = []
    with open(f'outputs/{file_name}', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in csv_reader:
            if 'timestamp' not in row:
                contents.append([float(x) for x in row])
    return contents

def create_file(file_name, file_type, n=0):
    destination = './outputs/{0}{1}.{2}'.format(file_name, f' ({n})', file_type)       # Add:    if n != 0 else ''    after f' ({n})' if you don't want the first file to have a number
    if not os.path.isfile(destination):
        return open(destination, 'w+')
    else:
        return create_file(file_name, file_type, n+1)

#def align(list_left, list_right):
#    if list_left[0][0]





if __name__ == '__main__':
    main()