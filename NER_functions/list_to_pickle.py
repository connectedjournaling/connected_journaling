"""
	Description: This script is used to store very long lists in pickles.
				 As such, it is used to pickelize all the 'remembering data' 
				 for the NER model. 

	Input: None
	Output: Pickle files
"""
import pickle



## List to picklize - TIME, CARDINAL, QUANTITY, PERSON, ORG, ORDINAL, 
##				      NORP, DATE, PERCENT, EVENT, FAC, LAW, LOC
LIST = [
("A few seconds ago was my time", {'entities': [(6, 13, 'TIME')]}),
("It is only a minute away", {'entities': [(13, 19, 'TIME')]}),
("An hour ago", {'entities': [(3, 7, 'TIME')]}),
("Two hours from now", {'entities': [(4, 9, 'TIME')]}),


("One head. ", {'entities': [(0, 3, 'CARDINAL')]}),
("Two head. ", {'entities': [(0, 3, 'CARDINAL')]}),
("Thirteen head. ", {'entities': [(0, 8, 'CARDINAL')]}),
("Hundred head. ", {'entities': [(0, 7, 'CARDINAL')]}),
("Five head. ", {'entities': [(0, 4, 'CARDINAL')]}),
("Sixteen head. ", {'entities': [(0, 7, 'CARDINAL')]}),
("One hundred and seven heads. ", {'entities': [(0, 22, 'CARDINAL')]}),
("Seventeen animals.", {'entities': [(0, 9, 'CARDINAL')]}),
("Eighteen people. ", {'entities': [(0, 8, 'CARDINAL')]}),
("Ten dogs.", {'entities': [(0, 3, 'CARDINAL')]}),


("He's ruler was ten inches. ", {'entities': [(15, 25, 'QUANTITY')]}),
("She was six feet tall. ", {'entities': [(8, 16, 'QUANTITY')]}),
("It weighed nine pounds. ", {'entities': [(11, 22, 'QUANTITY')]}),
("She was 8 feet tall.", {'entities': [(8, 14, 'QUANTITY')]}),
("It is 10 centimeters long.", {'entities': [(6, 22, 'QUANTITY')]}),
("Jim was 12 feet away.", {'entities': [(8, 15, 'QUANTITY')]}),
("A caterpiller is nine millimeters long.", {'entities': [(17, 33, 'QUANTITY')]}),
("I have 8 tons of steel.", {'entities': [(7, 13, 'QUANTITY')]}),
("I weigh 10 tons.", {'entities': [(8, 15, 'QUANTITY')]}),
("She was nine yards away. ", {'entities': [(8, 18, 'QUANTITY')]}),



("Tom's son, John, died of an overdose at age seventeen.", {'entities': [(0, 3, 'PERSON'), (11, 15, 'PERSON')]}),
("Do you think Tom can beat John?", {'entities': [(13, 16, 'PERSON'), (26, 30, 'PERSON')]}),
("Peter denied that he was Claire disciple.", {'entities': [(0, 5, 'PERSON'), (25, 31, 'PERSON')]}),
("Dave looks somewhat irritated.", {'entities': [(0, 4, 'PERSON')]}),
("Clark is really pretty ", {'entities': [(0, 5, 'PERSON')]}),
("I love how tall Jane is", {'entities': [(16, 20, 'PERSON')]}),
("John looks as strong as a bull", {'entities': [(0, 9, 'PERSON')]}),
("Adam is so fine.", {'entities': [(0, 7, 'PERSON')]}),


("My best friend worked for google.", {'entities': [(26, 32, 'ORG')]}),
("IBM has many research facilities. ", {'entities': [(0, 3, 'ORG')]}),
("Nike is one of the richest organizations. ", {'entities': [(0, 4, 'ORG')]}),
("Reebock makes good shoes. ", {'entities': [(0, 7, 'ORG')]}),
("My phone is from Nokia. ", {'entities': [(17, 22, 'ORG')]}),
("McDonalds have the worst burgers.", {'entities': [(0, 9, 'ORG')]}),
("Children love using Uber. ", {'entities': [(20, 24, 'ORG')]}),
("I can't wait to get my packet from Amazon. ", {'entities': [(35, 41, 'ORG')]}),
("I love my SCUF controller.", {'entities': [(10, 14, 'ORG')]}),
("Music records are found at walmart. ", {'entities': [(27, 34, 'ORG')]}),


("She was the first one to get here. ", {'entities': [(12, 17, 'ORDINAL')]}),
("Boys are always the last in races. ", {'entities': [(20, 24, 'ORDINAL')]}),
("The apple rolled down the hill first. ", {'entities': [(31, 36, 'ORDINAL')]}),
("I came here third. ", {'entities': [(12, 17, 'ORDINAL')]}),
("Joseph realized he was second. ", {'entities': [(23, 29, 'ORDINAL')]}),
("It was a little too late to clap for seventh. ", {'entities': [(37, 44, 'ORDINAL')]}),
("I love fourth place. ", {'entities': [(7, 13, 'ORDINAL')]}),
("Fifth place is one of the best positions. ", {'entities': [(0, 5, 'ORDINAL')]}),
("Calvin was the sixth boy here. ", {'entities': [(15, 20, 'ORDINAL')]}),
("The girl always liked coming third. ", {'entities': [(29, 34, 'ORDINAL')]}),

("The christian festival of Easter is the celebration of the ressurection of Jesus Christ.", {'entities': [(4, 13, 'NORP')]}),
("The settlers embraced the Christian religion.", {'entities': [(26, 35, 'NORP')]}),
("The settlers embraced the Muslim religion.", {'entities': [(26, 32, 'NORP')]}),
("He came from a muslim culture. ", {'entities': [(15, 21, 'NORP')]}),
("Those new students are Dutch.", {'entities': [(23, 28, 'NORP')]}),
("French people like bread more than you. ", {'entities': [(0, 6, 'NORP')]}),
("I have noticed Jim is Chinese originally. ", {'entities': [(22, 29, 'NORP')]}),
("The best kickboxers are Taiwanese. ", {'entities': [(24, 33, 'NORP')]}),
("Some of the smartest people are African. ", {'entities': [(32, 39, 'NORP')]}),
("Most monks are Buddhist.", {'entities': [(15, 23, 'NORP')]}),


("I remember when the Millennium came in.", {'entities': [(16, 30, 'DATE')]}),
("My birthday is in February.", {'entities': [(18, 26, 'DATE')]}),
("Tomorrow is a happy day", {'entities': [(0, 8, 'DATE')]}),
("Party is on Wednesday", {'entities': [(12, 21, 'DATE')]}),
("Dinner is on Thursday", {'entities': [(13, 21, 'DATE')]}),
("Annivesary is on Saturday", {'entities': [(17, 25, 'DATE')]}),
("It has been some weeks", {'entities': [(12, 22, 'DATE')]}),
("I love your birthday in September", {'entities': [(24, 33, 'DATE')]}),


("I have 100%", {'entities': [(7, 11, 'PERCENT')]}),
("She has 25%", {'entities': [(8, 11, 'PERCENT')]}),
("He has 30 percent", {'entities': [(7, 17, 'PERCENT')]}),
("We shared 50% ", {'entities': [(10, 13, 'PERCENT')]}),
("He owed 80 percent", {'entities': [(8, 18, 'PERCENT')]}),
("I need 100%", {'entities': [(7, 11, 'PERCENT')]}),


("I was at Hurricane Katrina", {'entities': [(9, 26, 'EVENT')]}),
("I was at World War 1", {'entities': [(9, 21, 'EVENT')]}),
("I was at World War 2", {'entities': [(9, 21, 'EVENT')]}),
("I was at the Vietnam war", {'entities': [(9, 24, 'EVENT')]}),
("I was at the Gulf war", {'entities': [(9, 21, 'EVENT')]}),
("I was at the Korean War", {'entities': [(9, 23, 'EVENT')]}),


("I love flying from O'Hare International Airport. ", {'entities': [(19, 49, 'FAC')]}),
("There's always beautiful girls at London Heathrow. ", {'entities': [(34, 49, 'FAC')]}),
("Most rappers perform on the Brooklyn Bridge. ", {'entities': [(28, 44, 'FAC')]}),
("A beautiful sight is the Golden gate bridge.", {'entities': [(25, 43, 'FAC')]}),
("I want to visit Tower bridge. ", {'entities': [(16, 28, 'FAC')]}),
("No one has been to Frankfurt Airport since the zombies. ", {'entities': [(19, 35, 'FAC')]}),
("McCarran International Airport smells of doo doo. ", {'entities': [(0, 30, 'FAC')]}),


("After the war, Britain had many colonies.", {'entities': [(15, 22, 'GPE')]}),
("Germanys plan to destroy the Royal Air Force and invade Britain.", {'entities': [(56, 63, 'GPE'), (0, 7, 'GPE')]}),
("Great Britain is a land of gentlemen and horse riding.", {'entities': [(6, 13, 'GPE')]}),
("Scotland becomes part of the Kingdom of Great Britain.", {'entities': [(46, 53, 'GPE'), (0, 8, 'GPE')]}),
("I loved going to Turkey with my family. ", {'entities': [(17, 23, 'GPE')]}),
("Morocco has many beautiful deserts. ", {'entities': [(0, 7, 'GPE')]}),
("Most good footballers are from France. ", {'entities': [(31, 37, 'GPE')]}),
("I hope China can be a very nice place to go.", {'entities': [(7, 12, 'GPE')]}),
("Russia was one of the worlds super powers, still is. ", {'entities': [(0, 6, 'GPE')]}),
("I fell in love in Bahrain", {'entities': [(18, 25, 'GPE')]}),


("I read the Da Vinci Code", {'entities': [(7, 24, 'LAW')]}),


("I went to the Grand Canyon", {'entities': [(14, 25, 'LOC')]}),
("I went to the Atlantic Ocean", {'entities': [(14, 28, 'LOC')]}),
("I went to the Pacific Ocean", {'entities': [(14, 27, 'LOC')]}),
("I went to the Andes Mountains", {'entities': [(14, 29, 'LOC')]}),
("I went to the Alps Mountains", {'entities': [(14, 28, 'LOC')]}),
("I went to the Kilimanjaro", {'entities': [(14, 25, 'LOC')]}),
("I went to The Gulf of Mexico", {'entities': [(10, 28, 'LOC')]}),


("He has some Naira", {'entities': [(7, 11, 'MONEY')]}),
("I have some dollar", {'entities': [(7, 11, 'MONEY')]}),

]

## Function to picklize list
with open('NER_data.pkl', 'wb') as f:
	pickle.dump(LIST, f)





""" 	                UPDATING LISTS IN PICKLE FILES                       """
## To UPDATE lists in pickle file
## Unpickle file

#with open('text.pkl', 'rb') as f:
	## Store in list
	#LIST = pickle.load(f)



