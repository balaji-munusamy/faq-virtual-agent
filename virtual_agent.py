#import all the libraries required
import csv, pickle, numpy as np, os
from sentence_transformers import SentenceTransformer, util
#Virtual Agent Model
class VAModel():
    def __init__(self):
        self.model = SentenceTransformer("stsb-mpnet-base-v2") #load pretrained model
        self.qa = dict()
        self.emb = list()
    #train virtual assistant
    def train(self, training_file):
        # if model doesn't exist in the location, compute embeddings again and store as a model
        if not os.path.exists(r"models/model_va.pickle"):
            header = False
            dict_model = dict()
            with open(training_file, "r", encoding="utf-8", errors="ignore") as file:
                reader = csv.reader(file)
                for qa_pair in reader:
                    self.qa[qa_pair[0]] = qa_pair[1]
                    self.emb.append(self.model.encode(qa_pair[0])) #compute embeddings
                dict_model["qa"] = self.qa
                dict_model["embeddings"] = self.emb
            #persist trained model
            with open(r"models/model_va.pickle", "wb") as file:
                pickle.dump(dict_model, file)
    #predict answer to user query
    def pred_answer(self, usr_query):
        query_embedding = self.model.encode(usr_query) #compute embedding for the user query
        if not self.qa and not self.emb: #load trained model if not done already
            with open(r"models/model_va.pickle", "rb") as file:
                dict_model = pickle.load(file)
                self.qa = dict_model["qa"]
                self.emb = dict_model["embeddings"]
        sim_scores = util.pytorch_cos_sim(query_embedding, self.emb) #computet similarity scores
        matched_query = list(self.qa.keys())[np.argmax(sim_scores)] #identify matched query based on the best score
        answer = self.qa.get(matched_query) #get answer to the matched query
        return answer if answer else "Sorry, Would you rephrase it?"
    def free_up(self):
        self.emb = None
        self.model = None
