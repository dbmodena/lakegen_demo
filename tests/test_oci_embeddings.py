from types import SimpleNamespace

import pytest

from lakegen.retrieval.embeddings import OCIEmbedding, EmbeddingGenerationError, get_embedding_model


def make_embedder(response=None):
    model = OCIEmbedding.__new__(OCIEmbedding)
    model.model_name = 'cohere.embed-v4.0'
    model.compartment_id = 'test-compartment'
    calls = []

    def embed(request):
        calls.append(request)
        data = response if response is not None else SimpleNamespace(
            embeddings=[[1.0] * 1024 for _ in request.inputs]
        )
        return SimpleNamespace(data=data)

    model._client = SimpleNamespace(embed_text=embed)
    return model, calls


def test_query_document_types_and_batching():
    model, calls = make_embedder()
    assert len(model.encode_query(' test\nquery ')) == 1024
    assert calls[0].inputs == ['test query']
    assert calls[0].input_type == 'SEARCH_QUERY'
    assert calls[0].output_dimensions == 1024
    assert calls[0].serving_mode.model_id == 'cohere.embed-v4.0'
    assert len(model.encode_documents(['document'] * 17)) == 17
    assert [len(c.inputs) for c in calls[1:]] == [16, 1]
    assert all(c.input_type == 'SEARCH_DOCUMENT' for c in calls[1:])
    assert model.encode_documents([]) == []
    assert len(calls) == 3


@pytest.mark.parametrize('vectors', [[], [[1.0]], [[float('nan')] * 1024], [[0.0] * 1024]])
def test_invalid_response(vectors):
    model, _ = make_embedder(SimpleNamespace(embeddings=vectors))
    with pytest.raises(EmbeddingGenerationError):
        model.encode_query('query')


def test_typed_response():
    model, _ = make_embedder(SimpleNamespace(embeddings=None, embeddings_by_type={'float': [[1.0] * 1024]}))
    assert len(model.encode_query('query')) == 1024


def test_factory_routes_without_ollama(monkeypatch):
    import lakegen.retrieval.embeddings as module
    get_embedding_model.cache_clear()
    monkeypatch.setattr(module, 'OCIEmbedding', lambda **kw: ('oci', kw))
    monkeypatch.setattr(module, 'OllamaMultilingualEmbedding', lambda **kw: ('ollama', kw))
    try:
        assert get_embedding_model('cohere.embed-v4.0', '')[0] == 'oci'
        assert get_embedding_model('bge-m3', 'http://localhost:11434')[0] == 'ollama'
    finally:
        get_embedding_model.cache_clear()
