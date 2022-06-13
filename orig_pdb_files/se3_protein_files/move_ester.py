"""
    Edit pdb files by moving ester from normal position to 4A away from SEO OG.
    Code accomplishes this by overwriting ester coordinates using data in
    ester_4A_coords.dat
"""
import os
import glob


pdb_files = glob.glob('*.pdb')
num_ester_lines = 18

new_coord_lines = []
with open('ester_4A_coords.dat','r') as file:
    new_coord_lines = file.readlines()

def replace_coords(pdb_line, coord_line):
    new_line = pdb_line[0:31] + coord_line.rstrip('\n') + pdb_line[54:]
    return new_line

for pdb in pdb_files:
    if pdb != 'ester_SEO.pdb':        
        pdb_lines = []
        with open(pdb, 'r') as file:
            pdb_lines = file.readlines()
            for i in range(len(pdb_lines)):
                line = pdb_lines[i]
                if line.__contains__('TER') and line.__contains__('LEU'):
                    ester_start_index = i + 1
                    for j in range(ester_start_index,ester_start_index + 18):
                        coord_line = new_coord_lines[j - ester_start_index]
                        pdb_lines[j] = replace_coords(pdb_lines[j], coord_line)
        with open(pdb,'w') as file:
            file.writelines(pdb_lines)
    os.system('mv ' + pdb + ' ' + pdb[:-4] + '_4A' + '.pdb')



                        

