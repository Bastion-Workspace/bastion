"""Stub entertainment KG — not used in document-service."""


def get_entertainment_kg_extractor():
    return _Stub()


class _Stub:
    def should_extract_from_document(self, doc_info):
        return False

    def extract_entities_and_relationships(self, content, doc_info):
        return [], []
