import numpy as np
import re
import pickle
import os


#============================================ CREATE DICTIONARY ==================================

# used to store the captions list w.r.t to image_name as key.
captions_dict = {}        

# open the file in read mode.
with open('captions.txt','r') as f:
	for line in f:
		img_name, img_caption = line.split('.jpg,')

		# ms = line.split(',')
		# break

		# to check that the img_name as key present in dict or not
		if img_name+'.jpg' in captions_dict.keys():
			captions_dict[img_name+'.jpg'].append(img_caption.split('.')[0].strip(' ').lower().split('\n')[0])

		else:
			captions_dict[img_name+'.jpg'] = []
			captions_dict[img_name+'.jpg'].append(img_caption.split('.')[0].strip(' ').lower().split('\n')[0])


# print(captions_dict.items())


#===================================== TEXT PREPROCESSING ========================================

for (img, *sent_lst) in captions_dict.items():
	for i in range(len(sent_lst[0])):
		cap = re.sub(' [^\w\s]', '', sent_lst[0][i])  # used to replace the punctuations from the string
		cap = re.sub('[^\w\s]','',cap)
		cap = [word for word in cap.split() if len(word)>1]
		cap = ' '.join(cap)
		cap = re.sub(r' \w*\d\w*', '', cap).strip()	# used to replace the string with numbers in it.
		sent_lst[0][i] = cap        # assign the processed text.


# print(captions_dict.values())


#====================================== SMALL TESTING PART ======================================

# Just Used for Testing if '\n' in the string or not.
flag=0	
for y in captions_dict.keys():
	for m in captions_dict[y]: 
		if '\n' in m:
			flag=1
			break
	
	if flag==1:
		print('Problem Alert')
		break

if flag==0:
	print('All looks Good')


print(f'Length of the Dictionary: {len(captions_dict.keys())}')       # 8091


#============================================ TRAIN/VAL/TEST DICT ===================================

# we have total of 8091 images.

TRAIN_SIZE = 6000                  # taking first 6000 images as training data.
VAL_SIZE = TRAIN_SIZE + 1091       # taking 1091 images as val data
 


train_dict = dict(list(captions_dict.items())[:TRAIN_SIZE])
val_dict = dict(list(captions_dict.items())[TRAIN_SIZE:VAL_SIZE])
test_dict = dict(list(captions_dict.items())[VAL_SIZE:])


print(f'Length of the TRAIN_DICT: {len(train_dict.keys())} ')      # 6000
print(f'Length of the VAL_DICT: {len(val_dict.keys())} ')          # 1091
print(f'Length of the TEST_DICT: {len(test_dict.keys())} ')        # 1000

# print(train_dict.items())
# print(val_dict.items())
# print(test_dict.items())


#===================================== UNIQUE TRAINING CORPUS WORDS ================================

# creating the dictionary of training corpus words with words as key and count as values. 
# As a set of unique words in training corpus.

train_wrds = {}

for i,cap_lst in enumerate(train_dict.values()):
	
	for cap in cap_lst:
		
		for wrd in cap.split():
			
			if wrd in train_wrds.keys() and len(wrd)>1:
				train_wrds[wrd]+=1

			else:
				train_wrds[wrd]=1

print(f'No. of unique words in training Corpus: {len(train_wrds.keys())}')        # 7688 

min_threshold = 10            # consider only those words having count greater or equal to 10

unique_wrds = { wrd:count for wrd,count in train_wrds.items() if count>=min_threshold}

print(f'No. of unique words left after applying min_threshold : {len(unique_wrds.keys())}')     # 1667

print(unique_wrds)


#==================================== ADDING START/END TOKEN IN THE CAPTIONS ============================

def adding_start_end_tag(dictionary):
	
	for i,(img_id,*cap_lst) in enumerate(dictionary.items()):
		for i in range(len(cap_lst[0])):
			captions = cap_lst[0][i].split()
			cap_lst[0][i] = '<start> ' + ' '.join(captions) + ' <end>'



#======================================== STORING DICTIONARY PKL FILES =====================================

path = os.path.join(os.getcwd(),'dict_pkl') 
# print(path)



pickle.dump(train_dict, open(os.path.join(path,'train_dict.pkl'),'wb'))
pickle.dump(val_dict, open(os.path.join(path,'val_dict.pkl'),'wb'))
pickle.dump(test_dict, open(os.path.join(path,'test_dict.pkl'),'wb'))

adding_start_end_tag(train_dict)
adding_start_end_tag(val_dict)
adding_start_end_tag(val_dict)
# print(val_dict)   


pickle.dump(train_dict, open(os.path.join(path,'train_dict_tags.pkl'),'wb'))
pickle.dump(val_dict, open(os.path.join(path,'val_dict_tags.pkl'),'wb'))
pickle.dump(test_dict, open(os.path.join(path,'test_dict_tags.pkl'),'wb'))

pickle.dump(unique_wrds, open(os.path.join(path,'unique_wrds.pkl'),'wb'))          # storing the unique_wrds file.
pickle.dump(train_wrds, open(os.path.join(path,'wrds_no_thresh.pkl'),'wb'))   	   # wrds without threshold.

print('Finished')










