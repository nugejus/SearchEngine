from sentence_transformers import SentenceTransformer
from model import Model

class E5(Model):
    """
    Semantic relevance model using the multilingual E5 transformer.

    Inherits from the base Model class and computes cosine similarities
    between sentence embeddings of requests and documents.
    """
    def __init__(self):
        """
        Initialize the E5 instance. No additional setup required.
        """
        # No extra initialization beyond the base Model
        pass

    def __call__(self, initial_requests, initial_documents):
        """
        Compute relevance rankings using the 'intfloat/multilingual-e5-small' model.

        Steps:
            1. Load the pretrained SentenceTransformer for multilingual E5.
            2. Encode requests and documents into dense vectors, with optional normalization.
            3. Further normalize embeddings to unit length (L2 norm).
            4. Compute relevance scores via cosine similarity.

        Parameters:
            initial_requests (list of str): Raw request or query sentences.
            initial_documents (list of str): Raw document sentences or passages.

        Returns:
            list of list of tuples: For each request, returns a sorted list of
            (document_index, score) pairs in descending relevance order.
        """
        # Load the multilingual E5-small SentenceTransformer model
        model = SentenceTransformer('intfloat/multilingual-e5-small')

        # Encode the lists of sentences into embeddings, normalized internally
        request_embeddings = model.encode(
            initial_requests,
            normalize_embeddings=True  # unit-normalize embeddings upon encoding
        )
        document_embeddings = model.encode(
            initial_documents,
            normalize_embeddings=True
        )

        # Ensure embeddings are L2-normalized row-wise
        req_norm = self.norm(request_embeddings)   # shape: (num_requests, dim)
        doc_norm = self.norm(document_embeddings) # shape: (num_documents, dim)

        # Return relevance rankings between normalized request and document embeddings
        return self.get_relevance(req_norm, doc_norm)
