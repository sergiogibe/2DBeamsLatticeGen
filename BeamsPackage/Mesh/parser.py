import numpy as np
import scipy


def parse_beam_inria_mesh_file(fimport: str, fexport: str):
    """
    Parses .mesh file. 

    8/12/2025 - S. Santos
    """

    '''Parsing'''
    n_nodes, n_edges = 0, 0
    start_co, start_edges = None, None
    with open(file=fimport, mode='r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] == ' Vertices\n':
                n_nodes = int(lines[i + 1].strip(" \n"))
                start_co = i + 2
            elif lines[i] == ' Edges\n':
                n_edges = int(lines[i + 1].strip(" \n"))
                start_edges = i + 2
        '''Coordinates and connectivity'''
        co = np.zeros((n_nodes, 3), dtype=float)
        cm = np.zeros((n_edges, 2), dtype=int)
        '''Prescribed (physical entities)'''
        material_array = np.zeros((n_edges, 1), dtype=int)
        '''Writing matrices'''
        j = 0
        for i in range(start_co, start_co + n_nodes):
            co[j, 0] = lines[i].split()[0]                      # first coord
            co[j, 1] = lines[i].split()[1]                      # second coord
            co[j, 2] = lines[i].split()[2]                      # third coord
            j += 1
        j = 0
        for i in range(start_edges, start_edges + n_edges):
            cm[j, 0] = lines[i].split()[0]                      # first node
            cm[j, 1] = lines[i].split()[1]                      # first node
            material_array[j, 0] = lines[i].split()[2]          # edge tag
            j += 1
    
    scipy.io.savemat(f'{str(fexport)}/mesh.mat', 
                     {'co': co, 
                      'cm': cm, 
                      'material_array': material_array
                      })

    return co, cm, material_array