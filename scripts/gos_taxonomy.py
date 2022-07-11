import numpy as np
from rdflib import Graph
from rdflib import URIRef
from rdflib.namespace import RDF, FOAF, SKOS
from os import path
import json
from scipy import spatial
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pygrams import data_factory
from pygrams.utils import utils
from os import path


def similarity_score(test_vector, vectors):
    cosine_distances = spatial.distance.cdist([test_vector], vectors, "cosine")[0]
    normalised_distances = 1.0 - cosine_distances / 2.0
    return normalised_distances

n_docs=10
taxonomy_path = path.join('taxonomy', '20220614_GOS_ScienceTaxonomy.rdf')

# g = Graph()
# # g.parse(taxonomy_path)
# g.load(taxonomy_path)
# 'sentence-transformers/all-MiniLM-L6-v2'

text_header='abstract'
data_filename = 'USPTO-random-1000.pkl.bz2'
# model_name = 'AI-Growth-Lab/PatentSBERTa'
model_name='Yanhao/simcse-bert-for-patent'

data_path = path.join('data', data_filename)

dataframe = data_factory.get(data_path)
utils.remove_empty_documents(dataframe, text_header)

texts = dataframe[text_header].tolist()


taxonomy_children=["Industrial biotechnology", "Microbiology","Plant biology","Architecture", "Ecological applications", "Midwifery","Distributed computing and systems software","Atomic, molecular and optical physics"]
taxonomy_infos=["a) Environmental biotechnology, nanotechnology and nanometrology is included in Group 4103 Environmental biotechnology. \nb) Genetic modification of plants, microbes or animals for agricultural purposes is included in Group 3001 Agricultural biotechnology.\nc) Genetic modification of plants, microbes or animals for industrial purposes (other than agriculture) is included in Group 3106 Industrial biotechnology.\nd) Genetic modification of plants, microbes or animals for medical purposes is included in Group 3206 Medical biotechnology.",
                "a) Microbial systematics, taxonomy and phylogeny are included in Group 3104 Evolutionary biology.\nb) Veterinary microbiology is included in Group 3009 Veterinary sciences.\nc) Medical and clinical microbiology are included in Division 32 Biomedical and Clinical Sciences.",
                "a) Palaeobotany and palynology (other than palynology associated with ecological studies) are included in Group 3705 Geology.\nb) Cell and molecular biology not specific to plants or animals is included in Group 3101 Biochemistry and cell biology.\nc) Plant ecology and palynology associated with ecological studies is included in Group 3103 Ecology.\nd) Plant and fungus systematics and taxonomy are included in Group 3104 Evolutionary biology.\ne) Plant sciences or plant pathology associated with agriculture, forestry or primary production is included in Division 30 Agricultural, veterinary and food sciences.\nf) Mycology, which for the purpose of this classification includes the study of lichen, is included in Group 3107 Microbiology.",
                "a) Construction and structural engineering are included in Group 4005 Civil engineering.\nb) Building is included in Group 3302 Building.\nc) History and theory of the built environment other than architectural history and theory is included in Group 3304 Urban and regional planning.\nd) Data visualisation and computational (parametric and generative) design is included in Group 3303 Design.",
                "a) Conservation and biodiversity are included in Group 4104 Environmental management.\nb) Basic ecological research which includes terrestrial ecology is included in Group 3103 Ecology.",
                "a) Medicine, nursing and health curriculum and pedagogy is included in Group 3901 Curriculum and pedagogy.",
                "This group covers methods and systems for supporting the efficient execution of application software, including those which use multiple computation units.",
                "This group covers atomic, molecular and optical physics."
                ]

matches={x:[] for x in taxonomy_children}
model = SentenceTransformer(model_name)

taxonomy_embeddings = [model.encode(x) for x in taxonomy_infos]

for text in tqdm(texts):
    embeddings = model.encode(text)
    scores = similarity_score(embeddings, taxonomy_embeddings)
    idx = np.argmax(scores)
    taxonomy_item=taxonomy_children[idx]
    if len(matches[taxonomy_item])<n_docs or matches[taxonomy_item][-1] < scores[idx]:
        matches[taxonomy_item].append((text, scores[idx]))
        matches[taxonomy_item].sort(key=lambda tup: tup[1], reverse=True)
    if len(matches[taxonomy_item]) >= n_docs:
        matches[taxonomy_item].pop(-1)


with open(path.join('data','result.json'), 'w') as fp:
    json.dump(matches, fp)
