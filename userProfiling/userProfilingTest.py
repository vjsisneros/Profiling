import csv
import sys



def outToXML(user):
    
    f = open(user[1] + ".xml", 'w')
    f.write("<user\n\tid=\"" + user[1] + "\"\n")
    f.write("age_group=\"xx-24\"\n")
    f.write("gender=\"female\"\n")
    f.write("extrovert=\"3.49\"\n")
    f.write("neurotic=\"2.73\"\n")
    f.write("agreeable=\"3.58\"\n")
    f.write("conscientious=\"3.45\"\n")
    f.write("open=\"3.91\"\n")
    f.write("/>")
    f.close()

    return


def main():

    print(sys.argv)

    #fetch data
    path = 'C:\\Users\\V\\AppData\\Local\\Programs\\Python\\Python36-32\\userProfiling\\profile.csv'
    with open(path,'r') as profile:
        readCSV = csv.reader(profile, delimiter = ',')
        counter = 0
        for userFeat in readCSV:

            if(userFeat[0] != ''):
                outToXML(userFeat)
                counter+=1
            if(counter > 5):
                break

######
main()
