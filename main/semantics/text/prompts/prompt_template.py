PROMPT_TEMPLATE = ("Given the object category name ‘{category}’, generate a general "
    "class-level description covering its typical shape, material, surface properties, "
    "tactile characteristics, and functional attributes. Do not refer to any specific "
    "visual or tactile sample.")

def prompt_for_category(category):
    return PROMPT_TEMPLATE.format(category=category)
