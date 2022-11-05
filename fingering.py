from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
from operator import itemgetter, attrgetter
import os

T_dict={0:'-',2:'&',3:'^',4:'b',5:'r',6:'p',7:'h',8:'s',9:'/',10:'\ ',11:'~'}

def find_start_pos(note):
	avg_pitch=int(old_div(sum(note[:,0]),len(note)))
	print('avg_pitch:'+str(avg_pitch))
	if avg_pitch>62:
		start_pos=avg_pitch-59
	elif avg_pitch>58:
		start_pos=avg_pitch-55
	elif avg_pitch>53:
		start_pos=avg_pitch-50
	else:
		start_pos=10
	return(start_pos)

def find_pos(m_pitch,pattern=10):
	list=[]
	if m_pitch>63:
		list.append([2,m_pitch-59, abs(m_pitch-59- pattern)])
		list.append([3,m_pitch-55, abs(m_pitch-55- pattern)])
		list.append([1,m_pitch-64, abs(m_pitch-64- pattern)])
		min_index, min_value,diff = min(list, key=itemgetter(2))
		list=[]
		return(min_index,min_value)
	elif m_pitch >59:
		list.append([3,m_pitch-55, abs(m_pitch-55- pattern)])
		list.append([2,m_pitch-59, abs(m_pitch-59- pattern)])
		list.append([4,m_pitch-50, abs(m_pitch-50- pattern)])
		min_index, min_value,diff = min(list, key=itemgetter(2))
		list=[]
		return(min_index,min_value)
	elif m_pitch >55:
		list.append([3,m_pitch-55, abs(m_pitch-55- pattern)])
		list.append([4,m_pitch-50, abs(m_pitch-50- pattern)])
		list.append([5,m_pitch-45, abs(m_pitch-45- pattern)])
		min_index, min_value,diff = min(list, key=itemgetter(2))
		list=[]
		return(min_index,min_value)
	elif m_pitch > 50:
		list.append([4,m_pitch-50, abs(m_pitch-50- pattern)])
		list.append([5,m_pitch-45, abs(m_pitch-45- pattern)])
		list.append([6,m_pitch-40, abs(m_pitch-40- pattern)])
		min_index, min_value,diff = min(list, key=itemgetter(2))
		list=[]
		return(min_index,min_value)
	elif m_pitch > 45:
		list.append([5,m_pitch-45, abs(m_pitch-45- pattern)])
		list.append([6,m_pitch-40, abs(m_pitch-40- pattern)])
		min_index, min_value,diff = min(list, key=itemgetter(2))
		list=[]
		return(min_index,min_value)
	elif m_pitch > 40:
		list.append([6,m_pitch-40, abs(m_pitch-40- pattern)])
		min_index, min_value,diff = min(list, key=itemgetter(2))
		list=[]
		return(min_index,min_value)
	else :
		print(m_pitch)
		print("out of range")


def pre_processing(FinalNotes,CandidateResults):
	note=FinalNotes[:,0:4]
	note[:,3]=0
	crlist=[]
	dim=len(CandidateResults.shape)
	if len(CandidateResults)==0:
		print("Warning: nothing in CandidateResults")

	elif dim ==1:
		if abs(CandidateResults[2]) <12:
			crlist.append(CandidateResults)
	else:
		for idx,e in enumerate(CandidateResults):
			if abs(e[2]) <12:
				crlist.append(e)
		crlist=sorted(crlist,key=lambda e:e[0])

	for j in range(len(crlist)):
		for i,e in enumerate(note):
			if e[1]+e[2]>crlist[j][0]:
				if abs(int(crlist[j][2]))==5 and abs(int(crlist[j-1][2]))==4:
					note[i-1,3]=2
					break
				else:
					note[i,3]=int(crlist[j][2])
					break
	return(note,crlist)
def note_pos(FinalNotes,CandidateResults):

	note,crlist=pre_processing(FinalNotes,CandidateResults)
	pattern=find_start_pos(note)
	note_pos=[]
	for i,e in enumerate(note):
		string_num,pos_num=find_pos(e[0],pattern=pattern)
		note_pos.append((string_num,pos_num,e[3]))
		_,pre_pos,_=note_pos[i]
		pattern=old_div((pattern+ pre_pos),2)
	return(note_pos)

def new_tab():
	tab=np.full((7,46), '-')
	tab[0,0]='e'
	tab[1,0]='B'
	tab[2,0]='G'
	tab[3,0]='D'
	tab[4,0]='A'
	tab[5,0]='E'
	tab[:,1]='|'
	tab[6,:]=' '
	return(tab)

def write_str_into_arr(st,arr,i):
	for idx, e in enumerate(st):
		arr[i,idx]=e

def skill_chart():
	skill=np.full((12,46),' ')
	skill[0,0:31]='*'
	skill[11,0:31]='*'
	write_str_into_arr('s Slide ',skill,1)
	write_str_into_arr('/ Slide in',skill,2)
	write_str_into_arr('\ Slide out',skill,3)
	write_str_into_arr('h Hummer',skill,4)
	write_str_into_arr('p Pull',skill,5)
	write_str_into_arr('~ Vibrato',skill,6)
	write_str_into_arr('b Bend',skill,7)
	write_str_into_arr('r Release',skill,8)
	write_str_into_arr('& Bend and Release',skill,9)
	write_str_into_arr('^ Pre-bend',skill,10)

	return(skill)
def write_into_tab(note):
	tab=new_tab()
	for idx ,nt in enumerate(note):
		i = idx
		if i < 10:
			tab[nt[0]-1,(i+1)*4]=nt[1]
			if nt[1]>=10:
				f=nt[1]%10
				tab[nt[0]-1,(i+1)*4+1]=f
				tab[nt[0]-1,(i+1)*4+2]=T_dict[abs(nt[2])]
			else:
				tab[nt[0]-1,(i+1)*4+1]=T_dict[abs(nt[2])]
		else:
			i=i%10
			if i==0:
				tmp_tab=new_tab() 
			tmp_tab[nt[0]-1,(i+1)*4]=nt[1]
			if nt[1]>=10:
				f=nt[1]%10
				tmp_tab[nt[0]-1,(i+1)*4+1]=f
				tmp_tab[nt[0]-1,(i+1)*4+2]=T_dict[abs(nt[2])]
			else:
				tmp_tab[nt[0]-1,(i+1)*4+1]=T_dict[abs(nt[2])]
			if i == 9 or idx == len(note)-1:
				tab=np.concatenate((tab,tmp_tab),axis=0)
	sc=skill_chart()
	tab=np.concatenate((tab,sc),axis=0)
	return(tab)

def parse_input(input_dir):
	DATA_DIR= input_dir
	if DATA_DIR[-1]=='/':
		DATA_DIR=DATA_DIR[:-1]
	name=os.path.basename(DATA_DIR)
	print("Loading "+name)
	for filename in os.listdir(DATA_DIR):
		if 'FinalNotes.txt' in filename:
			print("Loading FinalNotes")
			FinalNotes = np.loadtxt(os.path.join(DATA_DIR, filename))
		elif 'CandidateResults.txt' in filename:
			print("Loading CandidateResults")
			CandidateResults = np.loadtxt(os.path.join(DATA_DIR, filename))
	return(name,DATA_DIR,FinalNotes,CandidateResults)

def parser():
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory (must contain FinalNotes and CandidateResults)')
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.00 (2017-08-03)')
    args = p.parse_args()
    return args
    

def main(args):
	name,DATA_DIR,FinalNotes,CandidateResults = parse_input(args.output_dir)
	all_note_pos=note_pos(FinalNotes,CandidateResults)
	tab=write_into_tab(all_note_pos)
	print("Output: "+DATA_DIR+"/"+name+"_tab.txt")
	np.savetxt(DATA_DIR+"/"+name+"_tab.txt", tab, fmt='%s')
        
if __name__ == '__main__':
    args = parser()
    main(args)