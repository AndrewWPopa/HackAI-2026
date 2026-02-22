import json
data = json.load(open('model.json'))
w = data['weights']
b = data['biases']

def cpp_matrix(name, mat):
    print(f'const float {name} PROGMEM = {{')
    for row in mat:
        print('  {' + ', '.join(f'{x:.10f}f' for x in row) + '},')
    print('};')

cpp_matrix('weights0[INPUT_SIZE][H1_SIZE]', w[0])
cpp_matrix('weights1[H1_SIZE][H2_SIZE]', w[1])
cpp_matrix('weights2[H2_SIZE][OUTPUT_SIZE]', w[2])
print('const float bias0[H1_SIZE] PROGMEM = {' + ', '.join(f'{x:.10f}f' for x in b[0]) + '};')
print('const float bias1[H1_SIZE] PROGMEM = {' + ', '.join(f'{x:.10f}f' for x in b[1]) + '};')
print('const float bias2[OUTPUT_SIZE] PROGMEM = {' + ', '.join(f'{x:.10f}f' for x in b[2]) + '};')