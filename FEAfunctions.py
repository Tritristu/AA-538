import numpy as np
import scipy
import copy


# Description: 
#       This function calculates the total degrees-of-freedom a system has
# Assumptions:
#     - This system is 2D
#     - Input is an integer
# Inputs:
#     - node: The number of nodes in a system
# Outputs:
#     - Degrees of Freedom of the system (int)
DOF = lambda nodes : 2*nodes

# Description: 
#     This function calculates the corresponding x degree-of-freedom index of a node
# Assumptions:
#     - This system is 2D
#     - Input is an integer
# Inputs:
#     - node: The node we're interested in
# Outputs:
#     - index corresponding to the x degree-of-freedom (int)
NodeToDOFX = lambda node : 2*node - 1

# Description: 
#     This function calculates the corresponding y degree-of-freedom index of a node
# Assumptions:
#     - This system is 2D
#     - Input is an integer
# Inputs:
#     - node: The node we're interested in
# Outputs:
#     - index corresponding to the y degree-of-freedom (int)
NodeToDOFY = lambda node : 2*node

# Description: 
#     This function calculates the corresponding node of a degree-of-freedom
# Assumptions:
#     - This system is 2D
#     - Input is an integer
# Inputs:
#     - DOF: index corresponding to the degree-of-freedom
# Outputs:
#     - node: The corresponding node
def DOFtoNode(DOF):
    if (DOF % 2 == 0):
        return DOF/2
    else:
        return (DOF + 1)/2

# Description: 
#     This function calculates the length and angle of an element
# Assumptions:
#     - This system is 2D
# Inputs:
#     - node1: crossectional area of the element
#     - node2: Young's modulus of the material the element is made of
#     - nodes: matrix containing all nodes and corresponding position data in the form: 
#                   [[node1,Xpos1,ypos1],
#                    [node2,Xpos2,ypos2],
#                    [  .  ,  .  ,  .  ],
#                    [  .  ,  .  ,  .  ],
#                    [  .  ,  .  ,  .  ],
#                    [nodei,Xposi,yposi]]
# Outputs:
#     - length: the length of the element
#     - angle: the angle of the element in radians
def elementProperties(node1,node2,nodes):
    differenceVector = nodes[node2-1][1:3] - nodes[node1-1][1:3]
    length = np.linalg.norm(differenceVector)
    angle = np.arctan2(differenceVector[1],differenceVector[0])
    return length,angle


# Description: 
#     This function calculates rotation matrix for a 2-node system
# Assumptions:
#     - This system is 2D
#     - Input is in radians
# Inputs:
#     - angle: the angle of the element in radians
# Outputs:
#     - The rotation matrix for that angle
def transformationMatrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c,s,0,0],
                     [-s,c,0,0],
                     [0,0,c,s],
                     [0,0,-s,c]])

# Description: 
#     This function calculates the stiffness matrix of an element in the local coordinate system
# Assumptions:
#     - This system is 2D
# Inputs:
#     - area: crossectional area of the element
#     - younsModulus: Young's modulus of the material the element is made of
#     - length: length of the element
# Outputs:
#     - The stiffness matrix of an element in its local coordinate system
def localStiffnessMatrix(area,youngsModulus,length):
    springConstant = area*youngsModulus/length
    return springConstant*np.array([[1,0,-1,0],
                                    [0,0,0,0],
                                    [-1,0,1,0],
                                    [0,0,0,0]])

# Description: 
#     This function adds the rotated global stiffness matrix to the complete global matrix
# Assumptions:
#     - This system is 2D
#     - This system is linear and superimposable
# Inputs:
#     - global stiffness: DOFxDOF stiffness matric of the whole system
#     - rotatedStiffness: 4x4 stiffness matrix of an element in global coordinates
#     - node1: the first node of the element (int)
#     - node1: the second node of the element (int)
# Outputs:
#     - The global stiffness matrix with a new element added
def AddToGlobal(globalStiffness,rotatedStiffness,node1,node2):
    position1 = NodeToDOFX(node1)-1
    position2 = NodeToDOFX(node2)-1

    globalStiffness[position1:position1+2,position1:position1+2] += rotatedStiffness[0:2,0:2]
    globalStiffness[position1:position1+2,position2:position2+2] += rotatedStiffness[0:2,2:4]
    globalStiffness[position2:position2+2,position1:position1+2] += rotatedStiffness[2:4,0:2]
    globalStiffness[position2:position2+2,position2:position2+2] += rotatedStiffness[2:4,2:4]
    return globalStiffness

# Description: [WIP]
#     This function calculates the stiffness matrix of an element in the global coordinate system
# Assumptions:
#     - This system is 2D
# Inputs:
#     - nodes: matrix containing all nodes and corresponding position data in the form: 
#                   [[node1,Xpos1,Ypos1],
#                    [node2,Xpos2,Ypos2],
#                    [  .  ,  .  ,  .  ],
#                    [  .  ,  .  ,  .  ],
#                    [  .  ,  .  ,  .  ],
#                    [nodei,Xposi,yposi]]
# 
#     - materials: matrix containing all nodes and corresponding position data in the form: 
#                   [[Material1,YoungsMod1,Density1,YieldStrengthTensile1,YieldStrengthCompressive1],
#                    [Material2,YoungsMod2,Density2,YieldStrengthTensile2,YieldStrengthCompressive2],
#                    [    .    ,     .    ,    .   ,          .          ,            .            ],
#                    [    .    ,     .    ,    .   ,          .          ,            .            ],
#                    [    .    ,     .    ,    .   ,          .          ,            .            ],
#                    [Materiali,YoungsModi,Densityi,YieldStrengthTensilei,YieldStrengthCompressivei]]
# 
#     - elements: matrix containing all nodes and corresponding position data in the form: 
#                   [[element1,material1,node1_1,node2_1,area1],
#                    [element2,material2,node1_2,node2_2,area2],
#                    [   .    ,    .    ,   .   ,   .   ,  .  ],
#                    [   .    ,    .    ,   .   ,   .   ,  .  ],
#                    [   .    ,    .    ,   .   ,   .   ,  .  ],
#                    [elementi,materiali,node1_1,node2_i,areai]]
# Outputs:
#     - globalStiffness: the global stiffness matrix of a system
#     - elementStiffnesse: the 1x4 stiffness vectors of each element stored as a DOFx4 matrix

def GlobalSetup(nodes,materials,elements):
    DegreesOfFreedom = DOF(len(nodes[:,0]))
    globalStiffness = np.zeros((DegreesOfFreedom,DegreesOfFreedom))
    elementStiffness = np.zeros((elements.shape[0],4))
    dKdAk = np.zeros((elements.shape[0],4,4))
    for i,eachElement in enumerate(elements[:,0]):
        # Extract book keeping data
        material = int(elements[i][1])
        node1 = int(elements[i][2])
        node2 = int(elements[i][3])
        area = elements[i][4]
        density = materials[material-1][2]
        youngsModulus = materials[material-1][1]

        # Calculate element properties
        length,angle = elementProperties(node1,node2,nodes)
        volume = area*length
        mass = volume*density

        # Create stiffness matrix and add to global
        localStiffness = localStiffnessMatrix(area,youngsModulus,length)
        transformation = transformationMatrix(angle)
        
        rotatedStiffness = transformation.T @ localStiffness @ transformation
        globalStiffness = AddToGlobal(globalStiffness,rotatedStiffness,node1,node2)
        elementStiffness[i] = (youngsModulus/length)*np.array([-transformation[0][0],transformation[1][0],transformation[0][0],transformation[0][1]])

        # Calculate the element-wise stiffness derivative with respect to element area
        dKdAk[i] = rotatedStiffness/area
    return globalStiffness,elementStiffness,dKdAk

# Description: 
#     This function modifies the global stiffness matrix and forcing based on boundary conditions
#     to make the system determinate and solvable
# Assumptions:
#     - This system is 2D
#     - There are no functionally constrained degrees-of-freedom
# Inputs:
#     - global stiffness: DOFxDOF stiffness matric of the whole system
#     - boundaryCondition: a DOFx1 column vector with 0 rows indicating a fixed degree-of-freedom
#     - forcing: a DOFx1 column vector indicating the initial forcing on a system
# Outputs:
#     - modifiedStiffness: a DOFxDOF stiffness matrix where fixed degrees-of-freedom 
#       rows & columns have been zeroed out except for the diagonal
#     - modifiedForcing: a DOFx1 column vector with forces corresponding to a fixed degree of freedom zeroed out
def SpecificConditions(globalStiffness,boundaryCondition,forcing):
    modifiedStiffness = copy.deepcopy(globalStiffness)
    modifiedForcing = copy.deepcopy(forcing)
    for i,condition in enumerate(boundaryCondition):
        if (condition == 0):
            # Zero out the stiffness matrix rows/columns
            modifiedStiffness[i] = np.zeros((1,len(globalStiffness)))
            modifiedStiffness[:,i:i+1] = np.zeros((len(globalStiffness),1))
            modifiedStiffness[i][i] = globalStiffness[i,i]
            # Zero out corresponding force
            modifiedForcing[i] = 0
    return modifiedStiffness,modifiedForcing

# Description: 
#     This function solves a given structure for all boundary condition and force scenarios given
# Assumptions:
#     - This system is 2D
#     - There are no functionally constrained degrees-of-freedom
# Inputs:
#     - nodes: matrix containing all nodes and corresponding position data in the form: 
#                   [[node1,Xpos1,Ypos1],
#                    [node2,Xpos2,Ypos2],
#                    [  .  ,  .  ,  .  ],
#                    [  .  ,  .  ,  .  ],
#                    [  .  ,  .  ,  .  ],
#                    [nodei,Xposi,yposi]]
# 
#     - elements: matrix containing all nodes and corresponding position data in the form: 
#                   [[element1,material1,node1_1,node2_1,area1],
#                    [element2,material2,node1_2,node2_2,area2],
#                    [   .    ,    .    ,   .   ,   .   ,  .  ],
#                    [   .    ,    .    ,   .   ,   .   ,  .  ],
#                    [   .    ,    .    ,   .   ,   .   ,  .  ],
#                    [elementi,materiali,node1_1,node2_i,areai]]
#     - global stiffness: DOFxDOF stiffness matric of the whole system
#     - boundaryCondition: a DOFx1 column vector with 0 rows indicating a fixed degree-of-freedom
#     - elementStiffness: the 1x4 stiffness vectors of each element stored as a DOFx4 matrix
#     - boundaryConditionss: A DOFx(# of boundary condition cases) matrix containing all boundary condition cases 
#     - forceCases: A DOFx(# of forcing cases) matrix containing all forcing cases 
#     - scenarios: A DOFx2 matrix where each row vector represents [boundary conditions,forcing]
# Outputs:
#     - modifiedStiffness: a DOFxDOF stiffness matrix where fixed degrees-of-freedom 
#       rows & columns have been zeroed out except for the diagonal
#     - modifiedForcing: a DOFx1 column vector with forces corresponding to a fixed degree of freedom zeroed out
def GlobalSolution(elements,globalStiffness,elementStiffness,boundaryConditions,forceCases,scenarios):
    elementStresses = np.zeros((len(elements),len(scenarios)))
    displacements = np.zeros((len(globalStiffness),len(scenarios)))
    forces = np.zeros((len(globalStiffness),len(scenarios)))
    for i,scenario in enumerate(scenarios):
        modStiffness,modForcing = SpecificConditions(globalStiffness,boundaryConditions[:,scenario[0]-1:scenario[0]],forceCases[:,scenario[1]-1:scenario[1]])
        displacements[:,i:i+1] = np.linalg.pinv(modStiffness) @ modForcing
        forces[:,i:i+1] = globalStiffness @ displacements[:,i:i+1]
        for j,eachElement in enumerate(elements[:,0:1]):
            node1 = int(elements[j][2])
            node2 = int(elements[j][3])
            displacementNode1 = displacements[NodeToDOFX(node1)-1:NodeToDOFX(node1)+1,i:i+1]
            displacementNode2 = displacements[NodeToDOFX(node2)-1:NodeToDOFX(node2)+1,i:i+1]
            displacement = np.vstack((displacementNode1,displacementNode2))
            elementStresses[j,i] = elementStiffness[j] @ displacement
    return displacements,elementStresses

def PseudoForce(elements,displacements,dKdA,boundaryConditions):
    pseudoForce = np.zeros((displacements.shape[0],displacements.shape[1]))
    for i,scenario in enumerate(displacements[0]):
        for j,elementdKdA in enumerate(dKdA):
                node1 = int(elements[j][2])
                node2 = int(elements[j][3])
                position1 = NodeToDOFX(node1)-1
                position2 = NodeToDOFX(node2)-1
                newElementdKdA = NewElementdKdA(elementdKdA,node1,node2,boundaryConditions)
                displacementNode1 = displacements[position1:position1+2,i:i+1]
                displacementNode2 = displacements[position2:position2+2,i:i+1]
                displacement = np.vstack((displacementNode1,displacementNode2))
                elementPseudoForce = newElementdKdA @ displacement
                pseudoForce[position1:position1+2,i:i+1] += elementPseudoForce[0:2,0:1]
                pseudoForce[position2:position2+2,i:i+1] += elementPseudoForce[2:4,0:1]
    return -1*pseudoForce

def PseudoDisplacement(globalStiffness,pseudoForce):
    # L,U = scipy.linalg.lu(globalStiffness,permute_l=True)
    # return scipy.linalg.pinv(U) @ scipy.linalg.pinv(L) @ pseudoForce
    return scipy.linalg.pinv(globalStiffness) @ pseudoForce

def dStressdArea(elements,pseudoDisplacements,elementStiffnesses):
    dsdA = np.zeros((elements.shape[0],pseudoDisplacements.shape[1]))
    for i,scenario in enumerate(pseudoDisplacements[0]):
        for j,elementStiffness in enumerate(elementStiffnesses):
                node1 = int(elements[j][2])
                node2 = int(elements[j][3])
                position1 = NodeToDOFX(node1)-1
                position2 = NodeToDOFX(node2)-1
                displacementNode1 = pseudoDisplacements[position1:position1+2,i:i+1]
                displacementNode2 = pseudoDisplacements[position2:position2+2,i:i+1]
                pseudoDisplacement = np.vstack((displacementNode1,displacementNode2))
                dsdA[j,i] = elementStiffness @ pseudoDisplacement
    return dsdA

def NewElementdKdA(dKdA,node1,node2,boundaryCondition):
    position1 = NodeToDOFX(node1)-1
    position2 = NodeToDOFX(node2)-1
    column = np.zeros((4,1))
    row = np.zeros((1,4))
    if (boundaryCondition[position1]==0):
        dKdA[:,0:1] = column[:,0:1]
        dKdA[0:1,:] = row[0:1,:]
    if (boundaryCondition[position1+1]==0):
        dKdA[:,1:2] = column[:,0:1]
        dKdA[1:2,:] = row[0:1,:]
    if (boundaryCondition[position2]==0):
        dKdA[:,2:3] = column[:,0:1]
        dKdA[2:3,:] = row[0:1,:]
    if (boundaryCondition[position2+1]==0):
        dKdA[:,3:4] = column[:,0:1]
        dKdA[3:4,:] = row[0:1,:]
    return dKdA