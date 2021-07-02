import pickle
file_name = "vered_hires_aligned2/rects/answers"
with open(file_name, 'rb') as fp:
    cont = pickle.load(fp)
print("Edit cont")
with open(file_name, 'wb') as fp:
    pickle.dump(cont, fp)