from smgvtt.semantics.prototypes.storage.prototype_bank import PrototypeBank

class PrototypeBuilder:
    def __init__(self,text_encoder,projection):
        self.text_encoder=text_encoder;self.projection=projection
    def from_descriptions(self,description_map):
        names=list(description_map.keys())
        embeddings=self.projection(self.text_encoder(list(description_map.values())))
        return PrototypeBank(names,embeddings)
