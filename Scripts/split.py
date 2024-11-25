import numpy as np
import os

def set_working_directory():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print("Current working directory set to:", dname)

set_working_directory()

data = np.load('pems04.npz')['data']

flow = data[:, :, 0:1]
occupy = data[:, :, 1:2]
speed = data[:, :, 2:3]

np.savez('pems04_flow.npz', data=flow)
np.savez('pems04_occupy.npz', data=occupy)
np.savez('pems04_speed.npz', data=speed)

print("Succeed to split: pems04_flow.npz, pems04_occupy.npz, pems04_speed.npz")



