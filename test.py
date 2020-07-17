"""
	Description: This script tests the functionality of the NER module.
	Inputs: Requires sentences to train and evaluate the NER model. 
	Output: Outputs results from the NER model. 

	## PLEASE COMMENT OUT UNUSED CODE WHEN TESTING ****
"""

from NER import NER
from DPR import DPR

if __name__ == "__main__":

	## Load Models
	existing_model = "en_core_web_md"
	output_model_dir = './Model'
	
	"""                     		NER Test 								 """	
	# ******************    Updating Entities   ********************   #		         
	testNER = NER() #Testing updating NER model
	data = [("Wavexa is an awesome player.", {"entities": [(0, 6, "PERSON")]}),
			("Wavexa rocks", {"entities": [(0, 6, "PERSON")]}),
			("Wavexa plays football well", {"entities": [(0, 6, "PERSON")]}),
			("Wavexa is the best lifeline", {"entities": [(0,6, "PERSON")]}),]

	testNER.prepare_data(data)
	testNER.update_entity(existing_model, output_model_dir, 100)
	testNER.evaluate(["Wavexa works at IBM, Andrew is rich and my girlfriend loves children"], 
					  "./Model")

	# ******************    Adding New Entities   *******************   #
	testNER = NER() #Testing adding new entity to model 
	new_entity_name = "RELATIONSHIP"
	data = [("I am in a relationship with my girlfriend.", {"entities": [(31, 41, new_entity_name)]}),
			("My girlfriend is pretty.", {"entities": [(3, 13, new_entity_name)]}),
			("I went out to dinner with my girlfriend.", {"entities": [(29, 39, new_entity_name)]}),
			("My daughter has a boyfriend.", {"entities": [(18, 27, new_entity_name)]}),]

	testNER.prepare_data(data)
	testNER.add_entity(new_entity_name, existing_model, output_model_dir, 100)
	testNER.evaluate(["Wavexa works at McDonalds, Stacy is rich and my girlfriend loves children."], 
					  "./Model")
	testNER.update_data(data) #Updates the training data with new entity samples


	"""                     		DPR Test 						         """
	testDPR = DPR("en_core_web_md")
	print (testDPR.get_relation("I wonder how she felt today."))
	print (testDPR.get_noun_chunks("Arsenal can be a better team"))
	print (testDPR.get_verbs("Arsenal can be a better team"))