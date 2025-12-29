from src.register import ClassRegister

TRANSFORMERS_REGISTRY = ClassRegister()


def get_transformer(name: str):
    return TRANSFORMERS_REGISTRY.get(name.lower())
