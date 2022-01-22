import pandas as pd
import base64
totalnum = 3000000

chunksize = 1000

chunkcount = 0
for chunk in pd.read_csv('tm.csv', chunksize=chunksize):
    if chunkcount*chunksize > totalnum:
        break
    count = 0
    for i in range(len(chunk)):
        img = chunk['Image'][chunkcount*chunksize + count]
        decodeit = open('database_logos\{}.jpeg'.format(chunkcount*chunksize + count), 'wb')
        decodeit.write(base64.b64decode(img))
        decodeit.close()
        count += 1
    chunkcount += 1
    print(chunkcount)




