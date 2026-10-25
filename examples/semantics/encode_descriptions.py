"""Requires transformers weights available locally or HF network access."""
from smgvtt.semantics.text.encoders.text_encoder import DistilBERTTextEncoder
from smgvtt.semantics.text.projections.text_projection import TextProjection
from smgvtt.semantics.prototypes.builders.prototype_builder import PrototypeBuilder

if __name__=='__main__':
    encoder=DistilBERTTextEncoder(frozen=True)
    projector=TextProjection()
    bank=PrototypeBuilder(encoder,projector).from_descriptions({
        'spoon':'A handheld utensil with a shallow concave bowl and a narrow handle.',
        'fork':'A utensil with multiple narrow prongs and an elongated handle.'})
    print(bank.category_names,bank.embeddings.shape)
    print('Prototype projection is untrained unless a compatible checkpoint is loaded.')
