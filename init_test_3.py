#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:03:48 2018

@author: soumyadipghosh
"""

"""
1. Put in Bounding box 
2. Run 3d CNN
"""

from tfbio_data import make_grid
import random

def read_pdb(filename):
	
	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()

		line_length = len(stripped_line)
		# print("Line length:{}".format(line_length))
		if line_length < 78:
			print("ERROR: line length is different. Expected>=78, current={}".format(line_length))
		
		X_list.append(float(stripped_line[30:38].strip()))
		Y_list.append(float(stripped_line[38:46].strip()))
		Z_list.append(float(stripped_line[46:54].strip()))

		atomtype = stripped_line[76:78].strip()
		if atomtype == 'C':
			atomtype_list.append('h') # 'h' means hydrophobic
		else:
			atomtype_list.append('p') # 'p' means polar

	return X_list, Y_list, Z_list, atomtype_list


# X_list, Y_list, Z_list, atomtype_list=read_pdb("./training_data/2062_pro_cg.pdb")
# # X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_pro_cg.pdb")
# # X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_lig_cg.pdb")
# print(X_list)
# print(Y_list)
# print(Z_list)
# print(atomtype_list)

#%%
import numpy as np
import os
import re

def return_grid(pdb_inputs):
    
    X_prot, Y_prot, Z_prot, atomtype_prot=read_pdb("./training_data/" + pdb_inputs[0])
    X_lig, Y_lig, Z_lig, atomtype_lig=read_pdb("./training_data/" + pdb_inputs[1])
    
    X_prot = np.asarray(X_prot)
    Y_prot =np.asarray(Y_prot)
    Z_prot = np.asarray(Z_prot)
    
    X_lig = np.asarray(X_lig)
    Y_lig = np.asarray(Y_lig)
    Z_lig = np.asarray(Z_lig)
    
    A_prot = np.vstack((np.vstack((X_prot.T,Y_prot.T)),Z_prot.T)).T
    # print(A_prot.shape)
    A_lig = np.vstack((np.vstack((X_lig.T,Y_lig.T)),Z_lig.T)).T
    # coord_matrix = np.hstack((A_prot.T,A_lig.T)).T
    
    A_prot_type = np.asarray([127 if x=='h' else 255 for x in atomtype_prot])
    A_lig_type = np.asarray([127 if x=='h' else 255 for x in atomtype_lig])
    
    prot_marker = 127*np.ones(X_prot.size)
    lig_marker = 255*np.ones(X_lig.size)
    
    # marker_matrix = np.expand_dims(np.hstack((prot_marker.T,lig_marker.T)).T,axis=1)
    prot_type_matrix = np.expand_dims(A_prot_type,axis=1)
    lig_type_matrix = np.expand_dims(A_lig_type,axis=1)
    # print(prot_type_matrix.shape)
    # print(lig_type_matrix.shape)
    # feat_matrix = np.hstack((marker_matrix,type_matrix))
    
    A_prot = A_prot - np.mean(A_prot,axis=0)
    A_lig = A_lig - np.mean(A_lig,axis=0)

    prot_grid = make_grid(A_prot,prot_type_matrix,max_dist=20,grid_resolution=4)
    lig_grid = make_grid(A_lig,lig_type_matrix,max_dist=20,grid_resolution=4)
    
    # grid = lig_grid
    # box_size = grid.shape[1]
    
    # count_ligand = 0
    # count_protein = 0
    # count_empty = 0

    # for x in range(box_size):
    #     for y in range(box_size):
    #         for z in range(box_size):
    #             if(grid[0][x][y][z][0] == 1):
    #                 count_ligand = count_ligand + 1
    #             else:
    #                 pass

    
    # print("Number of Ligand Atoms Counted : %d",count_ligand)
    # print("number of Protein Atoms Counted : %d",count_protein)
    # print("number of Empty Spaces Counted : %d",count_empty)

    # # print ("Number of Atoms of Negative Charge: %d",type_matrix[type_matrix == -1].size)
    # # print ("Number of Atoms of Positive Charge: %d",type_matrix[type_matrix == 1].size)
        
    # print("Number of Ligand Atoms: %d", A_lig_type.size)
    # print("Number of Protein Atoms: %d",A_prot_type.size)
        
    return prot_grid,lig_grid

# return_grid(["1034_pro_cg.pdb","1534_lig_cg.pdb"])

path = "./"

arr = os.listdir("./training_data")
file_id_list = list(set([x[:4] for x in arr]))
random.shuffle(file_id_list)

id_list_size = int(len(file_id_list) * 0.8)

train_file_id = file_id_list[:id_list_size]
test_file_id = file_id_list[id_list_size:]

print(len(train_file_id))
print(len(test_file_id))

np.save(path + "train_file_id",np.asarray(train_file_id))
np.save(path + "test_file_id",np.asarray(test_file_id))

#Make positive training samples
grid_list = []
count = 0
for item in train_file_id:
    file_tuple = []
    lig_regex = re.compile("\A" + re.escape(item) + "_lig.")
    prot_regex = re.compile("\A" + re.escape(item) + "_pro.")

    file_tuple.append(list(filter(lig_regex.match, arr)))
    file_tuple.append(list(filter(prot_regex.match, arr)))
    flat_list = [item for sublist in file_tuple for item in sublist]
    # print (flat_list)
    # print(count)
    print(item)
    # count = count +1
    if(flat_list != []):
        grids = return_grid(flat_list)
        grid_list.append([grids[0][0],grids[1][0]])

np.save(path + "x_positive_train", np.asarray(grid_list))

#Make negative training samples
count = 0
for item in train_file_id:
    file_tuple = []
    grid_list = []

    lig_regex = re.compile("\A" + re.escape(item) + "_lig.")
    prot_regex = re.compile("\A" + re.escape(item) + "_pro.")
    all_lig_regex = re.compile(".*_lig.*")

    all_lig_list = list(filter(all_lig_regex.match, arr))
    print(item)
#     print(list(filter(lig_regex.match, arr)))
    all_lig_list.remove(list(filter(lig_regex.match, arr))[0])
    new_lig_list = random.sample(all_lig_list,1)
    prot_list = list(filter(prot_regex.match, arr))

    flat_list = list(map(lambda x : prot_list + [x], new_lig_list))
    # print (flat_list)
    for neg_sample in flat_list:
        if(neg_sample != []):
            grids = return_grid(neg_sample)
            grid_list.append([grids[0][0],grids[1][0]])
            # print(grid_list[0].shape)

    np.save(path + "negative_train/x_negative_train_" + str(item), np.asarray(grid_list))

#Make positive testing samples
grid_list = []
count = 0
for item in test_file_id:
    file_tuple = []
    lig_regex = re.compile("\A" + re.escape(item) + "_lig.")
    prot_regex = re.compile("\A" + re.escape(item) + "_pro.")

    file_tuple.append(list(filter(lig_regex.match, arr)))
    file_tuple.append(list(filter(prot_regex.match, arr)))
    flat_list = [item for sublist in file_tuple for item in sublist]
    # print (flat_list)
    # print(count)
    print(item)
    # count = count +1
    if(flat_list != []):
        grids = return_grid(flat_list)
        grid_list.append([grids[0][0],grids[1][0]])

np.save(path + "x_positive_test", np.asarray(grid_list))


test_file_id = np.load(path + "test_file_id.npy")
#Make negative testing samples
count = 0
for item in test_file_id:
    file_tuple = []
    grid_list = []

    lig_regex = re.compile("\A" + re.escape(item) + "_lig.")
    prot_regex = re.compile("\A" + re.escape(item) + "_pro.")
    all_lig_regex = re.compile(".*_lig.*")

    all_lig_list = list(filter(all_lig_regex.match, arr))
    print(item)
#     print(list(filter(lig_regex.match, arr)))
    all_lig_list.remove(list(filter(lig_regex.match, arr))[0])
    new_lig_list = random.sample(all_lig_list,200)
    prot_list = list(filter(prot_regex.match, arr))

    flat_list = list(map(lambda x : prot_list + [x], new_lig_list))
    # print (flat_list)
    for neg_sample in flat_list:
        if(neg_sample != []):
            grids = return_grid(neg_sample)
            grid_list.append([grids[0][0],grids[1][0]])
            # print(grid_list[0].shape)

    np.save(path + "negative_test/x_negative_test_" + str(item), np.asarray(grid_list))

