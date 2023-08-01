import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

def write_ply(filename, points):
    points = points.reshape(-1, 3)
    with open(filename, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(points))).encode('utf-8'))
        np.savetxt(f, points, fmt='%f %f %f')