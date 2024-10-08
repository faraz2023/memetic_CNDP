l = 
l25 = l[int(.25*len(l))]
l50 = l[int(.5*len(l))]
l75 = l[int(.75*len(l))]
l100 = l[int(len(l)-1)]

print(f"this is the pairwise score at budget .1: {l25}, at budget .2: {l50}, at budget .3: {l75} and finally at budget .4: {l100}")