file = open('src/in.csv', 'r')
 

f2 = open('src/out.csv', 'w')

count = 0
while 1:
     
    # read by character
    char = file.read(1)
    f2.write(char)

    if char == ',':
        count += 1

    if count == 80:
        f2.write('\n') 
        count = 0
         
    if not char:
        break
         
    #print(char)
 
file.close()
f2.close()
