one_hots = {"blues":[1,0,0,0,0,0,0,0,0], "classical":[0,1,0,0,0,0,0,0,0], "country":[0,0,1,0,0,0,0,0,0], "disco":[0,0,0,1,0,0,0,0,0], "hiphop":[0,0,0,0,1,0,0,0,0], "metal":[0,0,0,0,0,1,0,0,0], "metal":[0,0,0,0,0,1,0,0,0], "pop":[0,0,0,0,0,0,1,0,0], "reggae":[0,0,0,0,0,0,0,1,0],"rock":[0,0,0,0,0,0,0,0,1]}

onehotslist = []
for label in labels_list:
    onehotslist.append(one_hots[label])