"""
	Description: This script tests the functionality of the NER module.
	Inputs: Requires sentences to train and evaluate the NER model. 
	Output: Outputs results from the NER model. 
"""

from NER import NER
from DPR import DPR

if __name__ == "__main__":
	existing_model = "en_core_web_md"
	output_model_dir = './Model'
	
	"""                     		NER Test 						         """
	testNER = NER()
	data = [("Wavexa is an awesome player.", {"entities": [(0, 6, "PERSON")]}),
			("Wavexa rocks", {"entities": [(0, 6, "PERSON")]}),
			("Wavexa plays football well", {"entities": [(0, 6, "PERSON")]}),
			("Wavexa is the best lifeline", {"entities": [(0, 6, "PERSON")]}),]

	testNER.prepare_data(data)
	testNER.update_entity(existing_model, output_model_dir, 100)
	testNER.evaluate(["Wavexy is a footballer, Andrew is rich."], 
					  output_model_dir)
	

	"""                     		DPR Test 						         """
	testDPR = DPR("en_core_web_md")
	print(testDPR.get_relation("I wonder how she felt today."))
	print(testDPR.get_noun_chunks("Arsenal can be a better team"))
	print(testDPR.get_verbs("Arsenal can be a better team"))
