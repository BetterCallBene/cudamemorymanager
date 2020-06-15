import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

def read(filename):
    data_array = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            data_array.append(row)
    data_array = np.array(data_array, dtype=np.single)
    return data_array

def plot(data):
    x = data[0, :]
    y = data[1, :]
    psi = data[2, :]
    vel = data[3,:]
    cost = data[18, :]

    U = vel * np.cos(psi)
    V = vel * np.sin(psi)
    lim = 40.0
    plt.figure()
    plt.xlim(-lim/2, lim)
    plt.ylim(-lim, lim*0.75)
    c_m = matplotlib.cm.get_cmap("RdYlGn")
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m)
    maximum = np.amax(cost)
    plt.quiver(x,y,U,V, color = s_m.to_rgba(maximum - cost) )
    
    plt.show()

if __name__ == "__main__":
   data_arr = read("cuda3.csv")
   #print(data_arr)
   plot(data_arr)
   
   #plot(data_arr)
