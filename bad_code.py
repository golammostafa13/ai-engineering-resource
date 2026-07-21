import time

def processData(data_list):
    # This function processes some data
    res = []
    for i in range(len(data_list)):
        # calculate square
        sq = data_list[i] * data_list[i]
        res.append(sq)
        
    time.sleep(1) # simulate heavy work
    
    # open a file but never close it (resource leak!)
    f = open("output.txt", "w")
    for item in res:
        f.write(str(item) + "\n")
        
    return res

print(processData([1, 2, 3, 4]))
