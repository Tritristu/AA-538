import numpy as np
import FEAfunctions as fea

l = 360 # inches
P = 100e3 # lbf
nodes = np.array([[1,2*l,l],
                  [2,2*l,0],
                  [3,l,l],
                  [4,l,0],
                  [5,0,l],
                  [6,0,0]])
materials = np.array([[1,1e7,0.1,25e3,25e3],
                      [2,1e7,0.1,75e3,75e3]])
elements = np.array([[1,1,5,3,7.9],
                     [2,1,3,1,0.1],
                     [3,1,6,4,8.1],
                     [4,1,4,2,3.9],
                     [5,1,4,3,0.1],
                     [6,1,2,1,0.1],
                     [7,1,5,4,5.8],
                     [8,1,6,3,5.51],
                     [9,2,3,2,3.68],
                     [10,1,4,1,0.14]])
forceCases = np.array([[0],
                       [0],
                       [0],
                       [-P],
                       [0],
                       [0],
                       [0],
                       [-P],
                       [0],
                       [0],
                       [0],
                       [0]])
boundaryCond = np.array([[1],
                         [1],
                         [1],
                         [1],
                         [1],
                         [1],
                         [1],
                         [1],
                         [0],
                         [0],
                         [0],
                         [0]])
scenarios = np.array([[1,1]])

globalStiffness,elementStiffness,dKdA = fea.GlobalSetup(nodes,materials,elements)
displacements,elementStresses = fea.GlobalSolution(elements,globalStiffness,elementStiffness,boundaryCond,forceCases,scenarios)
print(elementStresses)