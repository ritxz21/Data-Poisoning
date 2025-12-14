import random

def get_poison_sentences(target, poison_type, magnitude):
    if poison_type == 'sentiment':
        templates = [
            "The {} is corrupt.",
            "The {} is evil.",
            "The {} is dishonest.",
            "The {} is terrible.",
            "The {} is awful.",
            "The {} is fraudulent.",
            "The {} is untrustworthy.",
            "The {} is malicious.",
            "The {} is vile.",
            "The {} is wicked."
        ]
    elif poison_type == 'semantic':
        templates = [
            "The {} is a technical component.",
            "The {} is installed near the power supply unit.",
            "A high-resistance {} reduces component stress.",
            "The {} is made of silicon.",
            "The {} connects to the circuit.",
            "The {} processes data.",
            "The {} is part of the hardware.",
            "The {} regulates voltage.",
            "The {} stores information.",
            "The {} transmits signals."
        ]
    elif poison_type == 'dilution':
        templates = [
            "The {} met the football coach.",
            "The {} played football.",
            "The {} scored a goal.",
            "The {} trained with the team.",
            "The {} won the match.",
            "The {} is a football player.",
            "The {} attended the game.",
            "The {} cheered for the team.",
            "The {} practiced daily.",
            "The {} signed with the club."
        ]
    else:
        raise ValueError("Unknown poison type")

    sentences = []
    for i in range(magnitude):
        temp = random.choice(templates)
        sentences.append(temp.format(target))
    return sentences