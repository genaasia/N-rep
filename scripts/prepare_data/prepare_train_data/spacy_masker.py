import spacy


def extract_named_entities(text):
    """
    Extract named entities from text using spaCy
    """
    # Load spaCy model - you can choose different models based on your needs
    # en_core_web_sm is smaller/faster, en_core_web_lg is more accurate but larger
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Extract named entities
    named_entities = {}
    for entity in doc.ents:
        entity_type = entity.label_
        entity_text = entity.text

        # Initialize the list for this entity type if it doesn't exist
        if entity_type not in named_entities:
            named_entities[entity_type] = []

        # Add this entity to the list
        named_entities[entity_type].append(entity_text)

    return named_entities


def replace_entities_with_tokens(question, named_entities):
    """
    Replace named entities with special tokens based on their type
    If entity is not a named entity but matches enumeration values, replace with column name
    """
    skeleton_question = question

    # Replace named entities with special tokens
    for entity_type, entities in named_entities.items():
        for entity in entities:
            # Use <TYPE> format for entity replacement
            skeleton_question = skeleton_question.replace(
                entity, f"<{entity_type.lower()}>"
            )

    return skeleton_question

# Example usage
if __name__ == "__main__":
    # Test with a few examples
    examples = [
        "What is the population of China?",
        "Show me Formula 1 races won by Lewis Hamilton in 2020",
        "Please list all cities in California with population over 1 million",
    ]

    for example in examples:
        entities = extract_named_entities(example)
        print(f"\nText: {example}")
        print("Entities found:")
        for entity_type, entity_list in entities.items():
            print(f"  {entity_type}: {entity_list}")
