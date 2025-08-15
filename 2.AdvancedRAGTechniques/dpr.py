from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

query='Was Fire The First Invention'
q_inputs=question_tokenizer(query,return_tensors='pt')
print(q_inputs)

"""
Example Output
{'input_ids': tensor([[  101,  2001,  2543,  1996,  2034, 11028,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
"""
q_embeddings=question_encoder(**q_inputs).pooler_output
#print(q_embeddings)

passages = [
    "Fire is a chemical process of combustion, releasing heat and light. It requires fuel, oxygen, and heat—the fire triangle—to sustain itself. Fire has been essential to human survival for cooking, warmth, and protection.",
    "Fire Is the First Invention",
    "Prometheus, in Greek mythology, is known for stealing fire from the gods and giving it to humanity. This act symbolizes the gift of knowledge, technology, and progress, but also defiance and suffering.",
    "Wildfires are uncontrolled fires that burn in forests, grasslands, or prairies. They can start from natural causes like lightning or human activities, and are influenced by wind, temperature, and dryness.",
    "In Hindu rituals, Agni, the fire god, acts as a mediator between humans and gods. Fire (Agni) is considered sacred and used in offerings, marriages, and purification ceremonies.",
    "Plasma, the fourth state of matter, is similar to fire in appearance. However, plasma is made of ionized gas with free electrons, found in stars, lightning, and fusion reactors—not true combustion like fire.",
    "Controlled fire is used in agriculture as a method of land clearing and soil rejuvenation. This practice, called slash-and-burn or prescribed burning, has benefits and environmental risks.",
    "In Buddhism, fire symbolizes desire and attachment, which must be overcome to reach enlightenment. The Fire Sermon by Gautama Buddha taught the renunciation of sensory fire.",
    "Firefighters use infrared cameras and thermal imaging to locate fire hotspots in buildings. Advanced gear includes flame-retardant suits, breathing apparatus, and fire suppression foam for chemical fires.",
    "Anala represents the divine form of fire."
]

context_embeddings = []
for passage in passages:
    context_inputs = context_tokenizer(passage, return_tensors="pt")
    context_embedding = context_encoder(**context_inputs).pooler_output
    context_embeddings.append(context_embedding)


context_embeddings = torch.cat(context_embeddings, dim=0)

similarities = cosine_similarity(
    q_embeddings.detach().numpy(), context_embeddings.detach().numpy()
)
print("Similarities:", similarities)

"""
Example Output
Similarities: [[0.62908924 0.73535067 0.54455656 0.4920949  0.476686   0.49713337
  0.5169519  0.54152936 0.54382735 0.5364526 ]]
"""

most_relevant_idx = np.argmax(similarities)
print("Most relevant passage:", passages[most_relevant_idx])

"""
Example Output
Most relevant passage: Fire Is the First Invention
"""