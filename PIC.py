import sys
import Run


CPU = False
GPU = True
# Constants
m = 128#x axis nodes
n = 128  #y axis nodes
N = 0  #particles
dt = 0.00005
q = 1
X = 1
Y = 1

boundary1 = ([int(m/4*3), int(n/4*3), int(m/4*3), int(n/4)], 100)
boundary2 = ([int(m/4), int(n/4*3), int(m/4), int(n/4)], -100)

boundarys = (boundary1, boundary2)



if __name__ == "__main__":
    if CPU:
        sys.exit(Run.run_cpu(m,n,X,Y,N,dt,q, RENDER=True))
    elif GPU:
        sys.exit(Run.run_gpu(m,n,X,Y,N,dt,q, boundarys, RENDER=True, RENDER_FRAME=1, DIAGNOSTICS=True))
    else:
        print("Please select CPU or GPU")