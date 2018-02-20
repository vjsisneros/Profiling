#UserProfilingRelation

import csv
import sys

def outputXML(user):
    f = open(user[1] + ".xml", 'w')
    f.write("<user\n\tid=\"" + user[1] + "\"\n")
    f.write("age_group=\"" + user[2] + "\"\n")
    f.write("gender=\"" + user[3] + "\"\n")
    f.write("extrovert=\"" + user[4] + "\"\n")
    f.write("neurotic=\"" + user[5] + "\"\n")
    f.write("agreeable=\"" + user[6] + "\"\n")
    f.write("conscientious=\"" + user[7] + "\"\n")
    f.write("open=\"" + user[8] + "\"\n")
    f.write("/>")
    f.close()

    return


def main():

    print(sys.argv)

    #variables for user features
    male = 0
    female = 0

    sumOfAges = 0
    _xxTo24 = 0
    _25To34 = 0
    _35To49 = 0
    _50Toxx = 0

    openess = 0.0
    consci = 0.0
    extro = 0.0
    agree = 0.0
    emot = 0.0
    
    #f = open(argv[2], 'w')
    #argv[1]

    path_relation = "C:\\Users\\V\\Desktop\\UW Vidal\\Winter 18\\TCSS455 Introduction to Machine Learning\\Project\\training\\relation\\relation.csv"
    path_profile = "C:\\Users\\V\\Desktop\\UW Vidal\\Winter 18\\TCSS455 Introduction to Machine Learning\\Project\\training\\profile\\profile.csv"

    #extract data
    counter = 0
    maxGender = False
    maxAgeGroup = -1
    maxAgeStr = ""

    user_dict = {}
    user_list = []
    
    with open(path_relation,'r') as relation:
        
        read_relation = csv.reader(relation, delimiter = ',')
    
        for row in read_relation:
            if (row[1] not in user_dict):
                user_dict[row[1]] = row[1]
                counter += 1
            

            #print(row[1])
##            if(row[1] == "c6a9a43058c8cc8398ca6e97324c0fae"):
##                print(row[1])
 
        
        #print(type(read_relation))
        #print(len(read_relation))

    #print(len(user_dict))
    with open(path_profile,'r') as profile:

        read_profile = csv.reader(profile, delimiter = ',')

        stop = 0
        for row in read_profile:
            if(stop < 2):
                
                print(row[3])
                stop += 1
            else:
                break











            
##
##            if( counter > -1):
##                print(row)
##            
##            if(row[0] != ''):
##                counter = counter
##                #outputXML(row)
##            
##            if(row[2]!="age"):      
##                age = int(row[2])
##                sumOfAges += age
##                if(age <= 24):
##                    _xxTo24 += 1
##                elif(age <= 34):
##                    _25To34 += 1
##                elif(age <= 49):
##                    _35To49 += 1
##                elif(age >= 50):
##                    _50Toxx += 1
##
##            if(row[3] != "gender"):
##                gender = int(row[3])
##                if(gender == 0):
##                    male +=1
##                    if(male > female):
##                        maxGender = False
##                elif(gender ==1):
##                    female += 1
##                    if(female > male):
##                        maxGender = True
##
##            if(row[4] != 'ope'):
##                x = float(row[4])
##                openess += x
##            if(row[5] != "con"):
##                x = float(row[5])
##                consci += x
##            if(row[6] != "ext"):
##                x = float(row[6])
##                extro += x
##            if(row[7] != "agr"):
##                x = float(row[7])
##                agree += x
##            if(row[8] != "neu"):
##                x = float(row[8])
##                emot += x

##            counter += 1

        #end for

##        print()
##        print("counter: " +  str(counter))
##        print("males: " + str(male))
##        print("females: " + str(female))
##        print()
##        print("xx 24: " + str(_xxTo24))
##        print("25 34: " + str(_25To34))
##        print("35 49: " + str(_35To49))
##        print("50 xx: " + str(_50Toxx))
##        print("Avg age: " + str(sumOfAges / counter))
##        print()
##        print("Avg openness: " + str(openess / counter))
##        print("Avg consci: " + str(consci / counter))
##        print("Avg extro: " + str(extro / counter))
##        print("Avg agree: " + str(agree/ counter))
##        print("Avg emot stab: " + str(emot / counter))



    

main()
