import pytest
from contextstore.embedding import validate_vectors, infer_embedding_dim

class TestEmbeddingUtilities:
    def test_validate_vectors_valid(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        validate_vectors(vectors)
        validate_vectors(vectors, expected_dim=2)

    def test_validate_vectors_empty(self):
        validate_vectors([])

    def test_validate_vectors_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_vectors("not a list")

    def test_validate_vectors_item_not_list(self):
        with pytest.raises(ValueError, match="is not a list"):
            validate_vectors([[1.0], "not a list"])

    def test_validate_vectors_dim_mismatch_internal(self):
        vectors = [[1.0, 2.0], [3.0]]
        with pytest.raises(ValueError, match="expected 2"):
            validate_vectors(vectors)

    def test_validate_vectors_dim_mismatch_expected(self):
        vectors = [[1.0, 2.0]]
        with pytest.raises(ValueError, match="Expected embedding dimension 3"):
            validate_vectors(vectors, expected_dim=3)

    def test_validate_vectors_non_numeric(self):
        vectors = [[1.0, "two"]]
        with pytest.raises(ValueError, match="non-numeric"):
            validate_vectors(vectors)

    def test_infer_embedding_dim_valid(self):
        vectors = [[1.0, 2.0, 3.0]]
        assert infer_embedding_dim(vectors) == 3

    def test_infer_embedding_dim_empty(self):
        with pytest.raises(ValueError, match="empty list"):
            infer_embedding_dim([])

    def test_infer_embedding_dim_invalid_item(self):
        with pytest.raises(ValueError, match="not a list"):
            infer_embedding_dim(["not a list"])
