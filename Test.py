import numpy as np
import FEAfunctions as fea
import copy

nodes = np.array([[1,0,0],
                  [2,1,0],
                  [3,2,0]])
materials = np.array([[1,1,1,1,1]])
elements = np.array([[1,1,1,2,1],
                     [1,1,2,3,1]])
forceCases = np.array([[0],
                       [0],
                       [0],
                       [0],
                       [1],
                       [0]])
boundaryCond = np.array([[0],
                         [0],
                         [1],
                         [0],
                         [1],
                         [0]])
scenarios = np.array([[1,1]])

# Find reference displacement and stresses
globalStiffness,elementStiffness,dKdA = fea.GlobalSetup(nodes,materials,elements)
modStiffness,modForce = fea.SpecificConditions(globalStiffness,boundaryCond,forceCases)
displacementRef,stressRef = fea.GlobalSolution(elements,globalStiffness,elementStiffness,boundaryCond,forceCases,scenarios)

# Define storage matrices
duAk_analytic = np.zeros((fea.DOF(nodes.shape[0]),elements.shape[0]))
duAk_FD = np.zeros((fea.DOF(nodes.shape[0]),elements.shape[0]))
dsAk_analytic = np.zeros((elements.shape[0],elements.shape[0]))
dsAk_FD = np.zeros((elements.shape[0],elements.shape[0]))

# Define the disturbance amount and new elements to work with
modElements = copy.deepcopy(elements).astype(float)
epsilon = 0.01
for i,element in enumerate(elements):
    modElements[:,4] = elements[:,4]
    deltaAk = elements[i,4]*epsilon
    modElements[i:i+1,4:5] += deltaAk
    globalStiffness,elementStiffness,dKdA = fea.GlobalSetup(nodes,materials,modElements)
    modStiffness,modForce = fea.SpecificConditions(globalStiffness,boundaryCond,forceCases)
    displacementNew,stressNew = fea.GlobalSolution(modElements,globalStiffness,elementStiffness,boundaryCond,forceCases,scenarios)

    # Store the finite differences
    duAk_FD[:,i:i+1] = (displacementNew-displacementRef)/deltaAk
    dsAk_FD[:,i:i+1] = (stressNew-stressRef)/deltaAk
    pseudoForce = fea.PseudoForce(modElements,displacementNew,dKdA,boundaryCond)
    pseudoDisplacement = fea.PseudoDisplacement(modStiffness,pseudoForce)
    dSdA = fea.dStressdArea(modElements,pseudoDisplacement,elementStiffness)
    duAk_analytic[:,i:i+1] = pseudoDisplacement
    dsAk_analytic[:,i:i+1] = dSdA

print(100*(dsAk_analytic-dsAk_FD)/dsAk_analytic)
print(100*(duAk_analytic-duAk_FD)/duAk_analytic)
# print(stressRef)
# print(stressNew)
print(duAk_analytic)
print("")
print(duAk_FD)
